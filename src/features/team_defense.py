"""Per-game team defense aggregation from per-player OAA.

For each (game, opposing-hitter-handedness) we sum OAA across the eight
non-DH fielders in the opposing lineup, split into infield (1B/2B/3B/SS)
and outfield (LF/CF/RF) buckets — catchers are skipped (Savant doesn't
publish catcher OAA; their defensive value sits in framing / blocking
which aren't OAA-comparable). Handedness-split OAA columns
(``outs_above_average_rhh`` / ``_lhh``) are used so an outfielder strong
vs. lefties gets credit only when a LHB is at the plate.

Functions
---------
* ``load_oaa(year)`` — read ``data/oaa/oaa_<year>.parquet``.
* ``build_oaa_lookup(oaa_df, stand)`` — ``(player_id, position) -> oaa``
  for the requested handedness slice.
* ``infield_outfield_oaa(lineup_player_ids, lineup_positions, lookup)``
  — sum across the eight fielders.

The aggregator is positional: ``lineup_position`` tells us where each
slot ACTUALLY played in this game (boxscore-derived). DH/PH/PR/P slots
contribute 0 OAA.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_OAA_ROOT = Path("data/oaa")

# Ground-ball adjustment uses ONLY 1B/2B/3B/SS — the four infield
# positions that actually field grounders. Catcher is excluded by design:
# catchers virtually never field ground balls in fair territory and
# Savant doesn't publish OAA for catchers anyway (the equivalent stat
# is catcher framing / blocking runs, which is not OAA-comparable).
INFIELD_POSITIONS: set[str] = {"1B", "2B", "3B", "SS"}
OUTFIELD_POSITIONS: set[str] = {"LF", "CF", "RF"}

# Slots we skip entirely when summing team OAA: DH, pinch-hitter,
# pinch-runner, pitcher, and catcher (see note above).
SKIP_POSITIONS: set[str] = {"DH", "PH", "PR", "P", "C", None}


def load_oaa(year: int,
             root: Path | str = DEFAULT_OAA_ROOT) -> pd.DataFrame:
    """Read the ``oaa_{year}.parquet`` written by ``fetch_oaa``."""
    root = Path(root)
    p = root / f"oaa_{year}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run: python -m src.ingest.fetch_oaa --year {year}"
        )
    df = pd.read_parquet(p)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    return df


DEFAULT_TARGET_PERCENTILE: float = 0.80


def count_games_by_position(lineups_long: pd.DataFrame
                             ) -> dict[tuple[int, str], int]:
    """Count games started at each fielding position from a
    ``lineups_long_<year>.parquet``.

    Only fielding positions (1B/2B/3B/SS/LF/CF/RF) are counted —
    DH/PH/PR/P/C rows are skipped because they don't contribute to the
    defense aggregation. Returns ``{(player_id, position) -> games}``.
    """
    keep = INFIELD_POSITIONS | OUTFIELD_POSITIONS
    df = lineups_long[lineups_long["position"].isin(keep)]
    df = df.dropna(subset=["player_id"])
    grouped = df.groupby(["player_id", "position"], as_index=False)
    counts = grouped["game_id"].nunique()
    return {
        (int(r.player_id), str(r.position)): int(r.game_id)
        for r in counts.itertuples(index=False)
    }


def _per_position_game_targets(
    games_by_pp: dict[tuple[int, str], int],
    percentile: float = DEFAULT_TARGET_PERCENTILE,
) -> dict[str, float]:
    """``{position -> target games}`` — the ``percentile``-th percentile
    of games started at each position across all players.

    Default percentile (0.80) lands roughly where "regular starter"
    begins for each position: ~50 games for 1B, ~70 for SS, ~33 for LF,
    etc. Players at or above target keep their raw OAA; players below
    get OAA scaled up by ``target / their_games``.
    """
    by_pos: dict[str, list[int]] = {}
    for (_, pos), g in games_by_pp.items():
        by_pos.setdefault(pos, []).append(g)
    return {
        pos: float(pd.Series(values).quantile(percentile))
        for pos, values in by_pos.items()
    }


def build_oaa_lookup(
    oaa_df: pd.DataFrame,
    stand: str | None = None,
    *,
    games_by_player_position: dict[tuple[int, str], int] | None = None,
    target_percentile: float = DEFAULT_TARGET_PERCENTILE,
    scale_to_full_season: bool = True,
) -> dict[tuple[int, str], float]:
    """``(player_id, position_abbrev) -> scaled_oaa`` for one handedness slice.

    ``stand`` can be ``'R'``, ``'L'``, or ``None`` (use total OAA).
    Missing values default to 0.0 when looked up via ``.get(..., 0.0)``.

    Playing-time scaling
    --------------------
    The ``oaa_runs_per_unit`` constant elsewhere in the matchup engine
    (0.0011) was calibrated to a FULL SEASON of OAA. Part-time players
    accumulate proportionally less OAA at the same per-game rate, so
    their season-end value reads "half-scale" if they played half the
    games. We scale up to a regular-starter-equivalent so the per-run
    formula applies the same coefficient to everyone:

    .. math::
        \\text{scaled\\_oaa} = \\text{raw\\_oaa} \\cdot \\max\\!\\bigg(1,\\ \\frac{\\text{target\\_games}}{\\text{games\\_played}}\\bigg)

    ``target_games`` is the ``target_percentile``-th percentile of
    games-started at that position in the season's lineups (default
    80th percentile — the threshold above which a player is a regular
    contributor). A backup with 25 games at 1B (target=50) gets scale
    = 2×. A starter with 130 games gets scale = 1× (no rescale, since
    they're already a "full-season-equivalent" reference).

    Pass ``games_by_player_position={}`` (or ``scale_to_full_season=False``)
    to skip scaling and use raw OAA.

    Players in ``oaa_df`` but missing from ``games_by_player_position``
    (e.g., they played fielding chances but never as a STARTER at that
    position) keep their raw OAA — the scaling can't run without a
    valid games count.
    """
    col_by_stand = {
        "R": "outs_above_average_rhh",
        "L": "outs_above_average_lhh",
        None: "outs_above_average",
    }
    col = col_by_stand.get(stand, "outs_above_average")

    targets: dict[str, float] = {}
    if scale_to_full_season and games_by_player_position:
        targets = _per_position_game_targets(
            games_by_player_position, percentile=target_percentile,
        )

    out: dict[tuple[int, str], float] = {}
    for _, row in oaa_df.iterrows():
        pid = row.get("player_id")
        pos = row.get("position_abbrev")
        if pd.isna(pid) or not pos:
            continue
        val = row.get(col)
        if pd.isna(val):
            continue
        raw_oaa = float(val)

        if targets:
            games = games_by_player_position.get((int(pid), str(pos)))
            target = targets.get(str(pos), 0.0)
            if games and target > 0:
                raw_oaa = raw_oaa * max(1.0, target / games)
            # else: no scaling — keep raw OAA

        out[(int(pid), pos)] = raw_oaa
    return out


def infield_outfield_oaa(
    lineup_player_ids: list[int] | None,
    lineup_positions: list[str | None] | None,
    lookup: dict[tuple[int, str], float],
) -> tuple[float, float]:
    """Sum OAA across the eight non-DH fielders.

    Returns ``(infield_oaa, outfield_oaa)``. DH/PH/PR/P/C slots contribute 0.
    A defender with no OAA row (rookie call-up, super-low-attempt player)
    also contributes 0 — the assumption is "league average until proven
    otherwise."

    Each defensive position (1B, 2B, 3B, SS, LF, CF, RF) can only be
    counted ONCE. If two players are tagged with the same position
    (rare in real boxscores, but happens routinely with projected
    lineups where two players share a modal position), the FIRST
    appearance in the batting order wins and subsequent ones contribute
    zero. This prevents double-counting a position when the projector
    couldn't disambiguate which player actually fields it tonight.
    """
    if lineup_player_ids is None or lineup_positions is None:
        return 0.0, 0.0
    inf = of = 0.0
    seen: set[str] = set()
    for pid, pos in zip(lineup_player_ids, lineup_positions):
        if pid is None or pos is None or pos in SKIP_POSITIONS:
            continue
        if pos in seen:
            continue
        seen.add(pos)
        try:
            key = (int(pid), pos)
        except (TypeError, ValueError):
            continue
        val = lookup.get(key, 0.0)
        if pos in INFIELD_POSITIONS:
            inf += val
        elif pos in OUTFIELD_POSITIONS:
            of += val
    return inf, of
