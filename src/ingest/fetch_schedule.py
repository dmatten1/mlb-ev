"""Pull MLB schedule + probable pitchers for one or more game dates.

The MLB-StatsAPI ``schedule`` endpoint with ``hydrate=probablePitcher``
returns each scheduled game's home/away probable pitcher ID (not just
name). We tidy that into a per-game DataFrame and persist a JSON
snapshot mirroring the ``fetch_outcomes`` layout:

    data/raw/schedule/baseball_mlb/<year>/<YYYY-MM-DD>.json

Probable-pitcher *handedness* isn't in this endpoint — we resolve it
from the local Statcast parquet (most recent ``p_throws`` per pitcher).
Pitchers with no Statcast appearances yet (call-ups making MLB debuts,
etc.) get ``hand=None`` and the lineup projector falls back to the
no-platoon pool.

CLI:
    # Today's schedule
    python -m src.ingest.fetch_schedule

    # Specific date
    python -m src.ingest.fetch_schedule --date 2026-05-18

    # Range
    python -m src.ingest.fetch_schedule --start 2026-05-18 --end 2026-05-20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "raw" / "schedule"
DEFAULT_STATCAST_ROOT = REPO_ROOT / "data" / "raw" / "statcast"

DEFAULT_SPORT_ID = 1  # MLB
SPORT_LABEL = "baseball_mlb"

logger = logging.getLogger("ingest.fetch_schedule")


# ---------------------------------------------------------------------------
# Schedule pull (StatsAPI)
# ---------------------------------------------------------------------------

def _safe(d: dict | None, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def fetch_schedule_for_date(target_date: date) -> list[dict[str, Any]]:
    """Pull a single date's schedule with probable pitchers resolved to IDs.

    Returns a list of tidy game dicts:
        game_id, game_date, game_datetime, game_type, status,
        home_id, home_name, away_id, away_name,
        home_probable_pitcher_id, home_probable_pitcher_name,
        away_probable_pitcher_id, away_probable_pitcher_name,
        venue_id, venue_name, doubleheader, game_num.
    """
    import statsapi

    raw = statsapi.get("schedule", {
        "sportId": DEFAULT_SPORT_ID,
        "date": target_date.isoformat(),
        "hydrate": "probablePitcher(note)",
    })
    dates = raw.get("dates") or []
    if not dates:
        return []
    games = dates[0].get("games") or []
    out: list[dict[str, Any]] = []
    for g in games:
        teams = g.get("teams") or {}
        home_t = teams.get("home") or {}
        away_t = teams.get("away") or {}
        home_pp = home_t.get("probablePitcher") or {}
        away_pp = away_t.get("probablePitcher") or {}
        out.append({
            "game_id": g.get("gamePk"),
            "game_date": (g.get("officialDate") or g.get("gameDate", "")[:10]),
            "game_datetime": g.get("gameDate"),
            "game_type": g.get("gameType"),
            "status": _safe(g, "status", "detailedState"),
            "abstract_status": _safe(g, "status", "abstractGameState"),
            "home_id": _safe(home_t, "team", "id"),
            "home_name": _safe(home_t, "team", "name"),
            "away_id": _safe(away_t, "team", "id"),
            "away_name": _safe(away_t, "team", "name"),
            "home_probable_pitcher_id": home_pp.get("id"),
            "home_probable_pitcher_name": home_pp.get("fullName"),
            "away_probable_pitcher_id": away_pp.get("id"),
            "away_probable_pitcher_name": away_pp.get("fullName"),
            "venue_id": _safe(g, "venue", "id"),
            "venue_name": _safe(g, "venue", "name"),
            "doubleheader": g.get("doubleHeader"),
            "game_num": g.get("gameNumber"),
        })
    return out


def build_schedule_record(games: list[dict[str, Any]],
                          *, target_date: date) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sport": SPORT_LABEL,
        "game_date": target_date.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "game_count": len(games),
        "data": games,
    }


# ---------------------------------------------------------------------------
# Probable-pitcher handedness resolver (from local statcast parquet)
# ---------------------------------------------------------------------------

def pitcher_hand_lookup(
    *,
    years: tuple[int, ...] | None = None,
    statcast_root: Path | str = DEFAULT_STATCAST_ROOT,
) -> dict[int, str]:
    """Build a ``{pitcher_id: 'R'|'L'}`` lookup from the local statcast parquets.

    Uses the *most recent* observed throwing hand across the supplied
    years. Pitchers can't change throwing arms, but the data sometimes
    has stray NaNs early-career; latest-non-null wins.
    """
    if years is None:
        years = (date.today().year,)
    statcast_root = Path(statcast_root)
    rows: list[pd.DataFrame] = []
    for y in years:
        p = statcast_root / f"statcast_{y}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["pitcher", "p_throws", "game_date"])
        rows.append(df)
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["pitcher", "p_throws"])
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce").astype("Int64")
    # Most-recent throws per pitcher.
    df = df.sort_values("game_date")
    latest = df.drop_duplicates("pitcher", keep="last")
    return dict(zip(latest["pitcher"].astype(int), latest["p_throws"]))


def attach_pitcher_hand(games: list[dict[str, Any]],
                        hand_map: dict[int, str]) -> list[dict[str, Any]]:
    """Add ``home_probable_pitcher_throws`` / ``away_probable_pitcher_throws``
    to each game dict by looking up each pitcher ID in ``hand_map``.
    """
    for g in games:
        for side in ("home", "away"):
            pid = g.get(f"{side}_probable_pitcher_id")
            try:
                pid_int = int(pid) if pid is not None else None
            except (TypeError, ValueError):
                pid_int = None
            g[f"{side}_probable_pitcher_throws"] = (
                hand_map.get(pid_int) if pid_int is not None else None
            )
    return games


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def schedule_relative_key(target_date: date, *, sport: str = SPORT_LABEL) -> str:
    return f"{sport}/{target_date.year}/{target_date.isoformat()}.json"


def write_local_schedule(record: dict[str, Any], *,
                         output_root: Path, target_date: date,
                         sport: str = SPORT_LABEL) -> Path:
    rel = schedule_relative_key(target_date, sport=sport)
    target = output_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    return target


def run_schedule_ingest(target_date: date, *,
                        local_root: Path = DEFAULT_LOCAL_ROOT,
                        years_for_hand: tuple[int, ...] | None = None,
                        statcast_root: Path = DEFAULT_STATCAST_ROOT,
                        skip_empty: bool = True) -> dict[str, Any]:
    """Pull, hand-attach, and persist one date's schedule."""
    games = fetch_schedule_for_date(target_date)
    years = years_for_hand or (target_date.year,)
    hand_map = pitcher_hand_lookup(years=years, statcast_root=statcast_root)
    games = attach_pitcher_hand(games, hand_map)
    record = build_schedule_record(games, target_date=target_date)
    destinations: list[str] = []
    if not (skip_empty and not games):
        path = write_local_schedule(record, output_root=local_root,
                                    target_date=target_date)
        destinations.append(str(path))
    return {
        "game_date": target_date.isoformat(),
        "game_count": len(games),
        "destinations": destinations,
    }


def run_backfill(start_date: date, end_date: date, **kwargs) -> dict[str, Any]:
    """Pull every date in ``[start_date, end_date]`` inclusive."""
    cur = start_date
    total = 0
    days_written = 0
    while cur <= end_date:
        s = run_schedule_ingest(cur, **kwargs)
        total += s["game_count"]
        if s["destinations"]:
            days_written += 1
        cur += timedelta(days=1)
    return {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "days_written": days_written,
        "total_games": total,
    }


# ---------------------------------------------------------------------------
# Load helper (for downstream feature builders)
# ---------------------------------------------------------------------------

def load_schedule_for_date(target_date: date | str,
                           *,
                           local_root: Path | str = DEFAULT_LOCAL_ROOT,
                           sport: str = SPORT_LABEL) -> pd.DataFrame:
    """Read the persisted JSON for ``target_date`` and return a tidy DataFrame.

    Empty DataFrame if the snapshot file doesn't exist yet.
    """
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    rel = schedule_relative_key(target_date, sport=sport)
    path = Path(local_root) / rel
    if not path.exists():
        return pd.DataFrame()
    record = json.loads(path.read_text(encoding="utf-8"))
    games = record.get("data") or []
    if not games:
        return pd.DataFrame()
    df = pd.DataFrame(games)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_datetime"] = pd.to_datetime(df["game_datetime"], errors="coerce", utc=True)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch MLB schedule + probable pitchers.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--date", help="Single ISO date (default: today).")
    grp.add_argument("--start", help="Start ISO date (requires --end).")
    p.add_argument("--end", help="End ISO date (requires --start).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    p.add_argument("--statcast-root", type=Path, default=DEFAULT_STATCAST_ROOT)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    if args.start:
        if not args.end:
            logger.error("--start requires --end.")
            return 2
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end)
        summary = run_backfill(s, e,
                               local_root=args.output_root,
                               statcast_root=args.statcast_root)
        logger.info("Backfill done: %s", summary)
    else:
        target = date.fromisoformat(args.date) if args.date else date.today()
        summary = run_schedule_ingest(target,
                                       local_root=args.output_root,
                                       statcast_root=args.statcast_root)
        logger.info("Wrote schedule for %s: %d games -> %s",
                    summary["game_date"], summary["game_count"],
                    summary["destinations"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
