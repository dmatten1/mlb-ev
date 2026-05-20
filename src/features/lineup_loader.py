"""Flatten raw boxscore JSON dumps into tidy per-game DataFrames.

Reads files written by ``src.ingest.fetch_boxscores`` and produces:

* A wide DataFrame with one row per game and columns for the 9 batters
  (home + away), both starters, and the lists of pitchers used.
* A "long" lineup DataFrame with one row per (game, side, batting_order_slot)
  for easier joining with hitter-level features.
* A player_id -> name mapping cache built from the same files.

The loader is offline — no network calls, no external API. It only reads
``data/raw/boxscores/<year>/<gamePk>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_BOXSCORE_ROOT = Path("data/raw/boxscores")
DEFAULT_OUTPUT_ROOT = Path("data/lineups")


def _box_paths(year: int | None, root: Path) -> list[Path]:
    if year is None:
        return sorted(root.rglob("*.json"))
    return sorted((root / str(year)).glob("*.json"))


def _player_positions(side: dict) -> dict[int, str]:
    """Build ``{player_id: starting_position_abbrev}`` for one side from
    the boxscore ``players`` dict.

    Each player record has ``allPositions`` (ordered list of every
    position they appeared at, in chronological order) and ``position``
    (the player's *last* appearance position). We want the **starting**
    position — the first entry of ``allPositions`` — because the lineup
    card we're predicting is the starting one, and downstream OAA /
    matchup features should be charged to the starter at each spot,
    not to the late-inning defensive sub.

    Why this matters: ``position.abbreviation`` reports the player's
    *final* position. A CF who shifts to LF in the 7th inning for a
    defensive replacement is recorded as an LF — wrong for predicting
    tomorrow's starting lineup. Using ``allPositions[0]`` fixes this.

    Falls back to ``position.abbreviation`` only if ``allPositions`` is
    missing (older boxscores or unusual game states).
    """
    out: dict[int, str] = {}
    players = side.get("players") or {}
    for key, p in players.items():
        try:
            pid = int(key.lstrip("ID")) if isinstance(key, str) else int(key)
        except (ValueError, AttributeError):
            continue
        all_pos = p.get("allPositions") or []
        pos: str | None = None
        if all_pos:
            first = all_pos[0] or {}
            pos = first.get("abbreviation")
        if not pos:
            pos = ((p.get("position") or {}).get("abbreviation"))
        if pos:
            out[pid] = pos
    return out


def parse_boxscore(doc: dict) -> dict:
    """Extract matchup essentials from one raw boxscore dict.

    Returns ``{}`` if essential fields are missing (corrupted / partial
    games). Callers should drop empty rows.

    Includes ``home_positions`` / ``away_positions`` as dicts mapping
    every player ID seen in that side's boxscore to the position they
    appeared at. Used downstream to look up per-player OAA for the
    defenders behind the opposing pitcher (DH excluded).
    """
    home = doc.get("home") or {}
    away = doc.get("away") or {}
    home_order = list(home.get("battingOrder") or [])
    away_order = list(away.get("battingOrder") or [])
    home_pitchers = list(home.get("pitchers") or [])
    away_pitchers = list(away.get("pitchers") or [])

    if not (home_order and away_order and home_pitchers and away_pitchers):
        return {}

    home_positions = _player_positions(home)
    away_positions = _player_positions(away)
    # Parallel-list form keyed by slot: ``positions[i]`` is the position
    # the slot-i+1 batter STARTED at. Parquet can't store dicts with
    # int keys, so we surface this list view too.
    home_lineup_positions = [home_positions.get(int(pid)) for pid in home_order]
    away_lineup_positions = [away_positions.get(int(pid)) for pid in away_order]

    return {
        "game_id_str": doc.get("gameId"),
        "home_team_id": (home.get("team") or {}).get("id"),
        "away_team_id": (away.get("team") or {}).get("id"),
        "home_starter_id": home_pitchers[0],
        "away_starter_id": away_pitchers[0],
        "home_lineup": home_order,
        "away_lineup": away_order,
        "home_lineup_positions": home_lineup_positions,
        "away_lineup_positions": away_lineup_positions,
        "home_pitchers_used": home_pitchers,
        "away_pitchers_used": away_pitchers,
        "home_bullpen_pregame": list(home.get("bullpen") or []),
        "away_bullpen_pregame": list(away.get("bullpen") or []),
    }


def _extract_player_names(doc: dict) -> dict[int, str]:
    """Pull ``{player_id: fullName}`` from one boxscore's ``players`` dict."""
    out: dict[int, str] = {}
    for side in ("home", "away"):
        players = (doc.get(side) or {}).get("players") or {}
        for key, p in players.items():
            try:
                pid = int(key.lstrip("ID")) if isinstance(key, str) else int(key)
            except (ValueError, AttributeError):
                continue
            name = (p.get("person") or {}).get("fullName")
            if name:
                out[pid] = name
    return out


def load_boxscores(
    year: int | None = None,
    root: Path | str = DEFAULT_BOXSCORE_ROOT,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Iterate boxscore JSONs and return ``(games_df, name_map)``.

    ``games_df`` is wide: one row per game, list-typed lineup columns.
    ``name_map`` maps every MLBAM ID seen to that player's full name.
    """
    root = Path(root)
    rows: list[dict] = []
    names: dict[int, str] = {}
    for path in _box_paths(year, root):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rec = parse_boxscore(doc)
        if not rec:
            continue
        gp = path.stem
        try:
            rec["game_id"] = int(gp)
        except ValueError:
            continue
        rec["box_path"] = str(path)
        rows.append(rec)
        names.update(_extract_player_names(doc))
    if not rows:
        return pd.DataFrame(), names

    df = pd.DataFrame(rows)
    # Stable column order
    front = [
        "game_id", "game_id_str",
        "home_team_id", "away_team_id",
        "home_starter_id", "away_starter_id",
        "home_lineup", "away_lineup",
        "home_lineup_positions", "away_lineup_positions",
        "home_pitchers_used", "away_pitchers_used",
        "home_bullpen_pregame", "away_bullpen_pregame",
    ]
    other = [c for c in df.columns if c not in front]
    df = df[front + other]
    return df, names


def explode_lineups(games_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format: one row per (game_id, side, batting_order_slot, player_id).

    ``slot`` is 1-9. Useful for joining hitter-level features keyed by
    ``player_id``.

    Also includes ``position`` (looked up from the per-side ``positions``
    dict in the wide frame), so the defense aggregator can apply
    OAA-by-position to the correct fielders.
    """
    pieces = []
    for side in ("home", "away"):
        lineup_col = f"{side}_lineup"
        pos_col = f"{side}_lineup_positions"
        has_pos = pos_col in games_df.columns
        keep = ["game_id", lineup_col]
        if has_pos:
            keep.append(pos_col)
        sub = games_df[keep].copy()
        # Explode lineup and positions together so their slot ordering aligns.
        if has_pos:
            sub_l = sub[["game_id", lineup_col]].explode(lineup_col, ignore_index=True)
            sub_p = sub[["game_id", pos_col]].explode(pos_col, ignore_index=True)
            sub = sub_l.assign(position=sub_p[pos_col])
        else:
            sub = sub.explode(lineup_col, ignore_index=True)
            sub["position"] = None
        sub["slot"] = sub.groupby("game_id").cumcount() + 1
        sub = sub.rename(columns={lineup_col: "player_id"})
        sub["side"] = side
        pieces.append(sub[["game_id", "side", "slot", "player_id", "position"]])
    out = pd.concat(pieces, ignore_index=True)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    return out


def rollup_to_parquet(
    year: int,
    *,
    boxscore_root: Path | str = DEFAULT_BOXSCORE_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, Path, Path]:
    """Read all ``<year>`` boxscores and write three parquets:

    * ``lineups_{year}.parquet``      — wide, one row per game (list columns).
    * ``lineups_long_{year}.parquet`` — one row per (game, side, slot).
    * ``player_names.parquet``        — id -> name (overwritten cumulatively).
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    games_df, names = load_boxscores(year=year, root=boxscore_root)

    wide_path = output_root / f"lineups_{year}.parquet"
    long_path = output_root / f"lineups_long_{year}.parquet"
    names_path = output_root / "player_names.parquet"

    games_df.to_parquet(wide_path, index=False)
    explode_lineups(games_df).to_parquet(long_path, index=False)

    # Merge with any existing names file so each rollup grows the cache.
    name_df = pd.DataFrame({"player_id": list(names.keys()),
                            "name": list(names.values())})
    if names_path.exists():
        existing = pd.read_parquet(names_path)
        name_df = (
            pd.concat([existing, name_df], ignore_index=True)
            .drop_duplicates("player_id", keep="last")
        )
    name_df.sort_values("player_id").to_parquet(names_path, index=False)

    return wide_path, long_path, names_path


def load_player_names(
    root: Path | str = DEFAULT_OUTPUT_ROOT,
) -> dict[int, str]:
    """Return the player_id -> name lookup. Empty dict if file missing."""
    p = Path(root) / "player_names.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(df["player_id"].astype(int), df["name"]))
