"""Project a team's likely batting order for a future game.

Strategy
--------
**Hungarian (max-weight bipartite matching) lineup assignment**,
platoon-split by opposing-starter handedness, IL-filtered to the
current 26-man active roster.

Procedure for projecting team ``T``'s lineup for a game on date ``D``
against an opposing starter of handedness ``H``:

1. Read the team's actual lineups from the last ``N`` days (boxscores
   via ``data/lineups/lineups_long_<year>.parquet`` joined with
   ``data/features/training_<year>.parquet``).
2. Annotate each game with the opposing starter's handedness.
3. Filter to games vs ``H`` ("same-platoon pool"). If that pool has
   fewer than ``MIN_POOL_GAMES`` rows, broaden to all recent games
   (no platoon filter).
4. **Filter to the current 26-man active roster** — drops IL'd /
   DFA'd / optioned players.
5. **Solve the lineup as a bipartite matching problem.** Build a cost
   matrix of (top-N most-active players × 9 positions) where cell
   ``[i, j]`` is the number of pool starts player ``i`` has at
   position ``j``. ``scipy.optimize.linear_sum_assignment`` returns
   the assignment that *maximizes total games-played across all nine
   (player, position) pairs*. This jointly chooses **which players
   make the lineup AND where each of them plays**, so an everyday
   rotation regular (Friedl rotating CF/LF) competes against a single-
   position specialist (Myers always at CF) on equal footing, and
   conditional patterns (when Hayes plays 3B, Stewart slides to 1B and
   Steer to RF) are recovered automatically.
6. **Order the chosen 9 players into a batting card** by their
   per-player mean batting slot in the pool. Leadoff candidate first,
   9-hole bat last.

Why Hungarian (vs greedy per-position)?
---------------------------------------
A naive greedy "for each position, pick the modal player there"
breaks on rotation regulars and conditional swaps:

- *Modal player at CF = Myers (13 starts).* Greedy picks him.
- *Friedl (12 total starts, split CF:6 / LF:6) gets squeezed out* even
  though he plays more games than Myers — just at varying positions.
- *Stewart and Steer tie at 1B (10 each).* Greedy tiebreak is arbitrary.
  In reality, when Hayes plays 3B, Stewart takes 1B and Steer slides
  to RF — the assignment is conditional on the rest of the lineup.

Solving the assignment jointly (Hungarian) picks the **highest-total-
weight 9-position lineup**, which naturally captures these patterns:
total-games-played dominates so rotation regulars stay in, and
position-specific games-played dominates inside the lineup to settle
who plays where.

Design choices
--------------
* **Two lookback windows** (asymmetric on purpose):
    * **All-platoon pool: 14 days.** Captures the team's most recent
      personnel decisions without stale rotations.
    * **Same-hand pool: 30 days.** Same-hand starts are rare —
      widening the platoon window doubles the sample so we can
      actually project a platoon-aware lineup against LHP.
* **MIN_POOL_GAMES = 3** same-hand starts (in the 30d window) before
  we believe the platoon pool; otherwise fall back to the 14d
  all-platoon pool.
* **Active-roster filter** (``filter_active_roster=True`` by default).
* **Position priority order**: ``C → SS → 2B → 3B → 1B → CF → LF →
  RF → DH``. Most-unique positions first (catchers, middle infield)
  so utility / corner-overlap players don't crowd out the regulars.

The projector is intentionally simple. v2 stretches: scrape
BaseballPress/Rotowire projected lineups for a sanity prior;
exponentially decay the weight on older games.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

LOOKBACK_DAYS: int = 14          # all-platoon pool window
PLATOON_LOOKBACK_DAYS: int = 30  # same-hand pool window (wider — same-hand starts are rarer)
MIN_POOL_GAMES: int = 3
LINEUP_SLOTS: int = 9

logger = logging.getLogger("features.lineup_projection")


# ---------------------------------------------------------------------------
# Active-roster (IL filter) lookup
# ---------------------------------------------------------------------------

def fetch_active_roster_ids(team_id: int) -> set[int]:
    """Return the set of MLBAM player IDs on the team's current 26-man active
    roster. IL'd, optioned, and DFA'd players are excluded.

    One ~100ms HTTP call. Callers should cache.
    """
    import statsapi

    try:
        data = statsapi.get("team_roster",
                             {"teamId": int(team_id), "rosterType": "active"})
    except Exception as e:  # noqa: BLE001
        logger.warning("[il-filter] roster fetch failed for team %s: %s "
                       "— falling back to no IL filter for this team.",
                       team_id, e)
        return set()
    out: set[int] = set()
    for p in (data.get("roster") or []):
        pid = (p.get("person") or {}).get("id")
        if pid is not None:
            out.add(int(pid))
    return out


def build_active_roster_map(team_ids: list[int]) -> dict[int, set[int]]:
    """Fetch each team's active roster, return ``{team_id: {player_id, ...}}``.

    Empty set means "no filter" (fetch failed or team unknown). Caller
    should treat an empty set as 'don't restrict'.
    """
    out: dict[int, set[int]] = {}
    for tid in sorted(set(int(t) for t in team_ids)):
        out[tid] = fetch_active_roster_ids(tid)
    return out


# ---------------------------------------------------------------------------
# Actual-lineup override (used when teams publish their lineup card)
# ---------------------------------------------------------------------------

@dataclass
class PublishedLineups:
    """The actual lineup card for one game, pulled from the boxscore API."""

    game_id: int
    home_lineup: list[int]
    away_lineup: list[int]
    home_positions: list[str | None]
    away_positions: list[str | None]
    home_starter_id: int | None
    away_starter_id: int | None


def _extract_lineup_from_boxscore(side: dict) -> tuple[list[int], list[str | None]]:
    """Pull batting order + positions from one side of a boxscore dict.

    Mirrors ``lineup_loader._player_positions`` + the slot ordering from
    ``parse_boxscore`` but operates on the live ``statsapi.boxscore_data``
    response shape rather than the saved JSON.
    """
    order = list(side.get("battingOrder") or [])
    players = side.get("players") or {}
    positions: list[str | None] = []
    for pid in order:
        key = f"ID{int(pid)}"
        p = players.get(key) or {}
        pos = ((p.get("position") or {}).get("abbreviation"))
        positions.append(pos)
    return [int(x) for x in order], positions


def fetch_published_lineup(game_id: int) -> PublishedLineups | None:
    """Try to pull an already-posted lineup card for one upcoming game.

    Returns ``None`` if either side hasn't posted yet (the standard
    pre-game state). When BOTH sides are posted, returns a
    :class:`PublishedLineups`.

    Implementation: ``statsapi.boxscore_data(game_id)`` returns the same
    ``home`` / ``away`` dicts that the boxscore JSON has, populated with
    ``battingOrder`` + ``players`` as soon as the team submits its
    lineup (~3 hours before first pitch).
    """
    import statsapi

    try:
        bs = statsapi.boxscore_data(int(game_id))
    except Exception as e:  # noqa: BLE001
        logger.debug("[override] boxscore_data failed for %s: %s", game_id, e)
        return None

    home_side = bs.get("home") or {}
    away_side = bs.get("away") or {}
    home_order, home_positions = _extract_lineup_from_boxscore(home_side)
    away_order, away_positions = _extract_lineup_from_boxscore(away_side)
    if len(home_order) < LINEUP_SLOTS or len(away_order) < LINEUP_SLOTS:
        return None

    # Starter IDs come from the per-side ``pitchers`` list (first entry).
    home_pitchers = list(home_side.get("pitchers") or [])
    away_pitchers = list(away_side.get("pitchers") or [])
    home_sp = int(home_pitchers[0]) if home_pitchers else None
    away_sp = int(away_pitchers[0]) if away_pitchers else None

    return PublishedLineups(
        game_id=int(game_id),
        home_lineup=home_order[:LINEUP_SLOTS],
        away_lineup=away_order[:LINEUP_SLOTS],
        home_positions=home_positions[:LINEUP_SLOTS],
        away_positions=away_positions[:LINEUP_SLOTS],
        home_starter_id=home_sp,
        away_starter_id=away_sp,
    )


FIELDING_POSITIONS: frozenset[str] = frozenset({"1B", "2B", "3B", "SS",
                                                   "LF", "CF", "RF"})

# Order in which we assign positions in the new position-first projector.
# Most-unique positions first (catchers, middle infielders) so utility
# guys with overlapping eligibility don't crowd out the regulars.
POSITION_PRIORITY: tuple[str, ...] = ("C", "SS", "2B", "3B", "1B",
                                        "CF", "LF", "RF", "DH")


@dataclass
class ProjectedLineup:
    """One team's projected (or actual) 9-batter lineup for a game.

    ``source`` is ``'projected'`` if the modal-mode estimator picked the
    lineup, or ``'actual'`` if it was pulled from MLB-StatsAPI's lineup
    card (typically posted ~3 hours before first pitch). For ``'actual'``
    rows, ``pool_size`` and ``used_platoon_split`` are not meaningful
    (set to 0 / False).
    """

    team_id: int
    as_of_date: date
    opposing_hand: str | None  # 'R', 'L', or None if unknown
    pool_size: int             # how many games went into the projection
    used_platoon_split: bool   # True = same-hand pool, False = all-platoon fallback
    player_ids: list[int]      # length 9, may contain placeholders if pool too thin
    positions: list[str | None]  # length 9, same order as player_ids
    source: str = "projected"  # 'projected' | 'actual'


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _load_history(year: int,
                  lineups_root: Path,
                  features_path: Path) -> pd.DataFrame:
    """Load and join lineups + opposing-SP handedness into one long frame:
    one row per (game, team, slot) with the opposing starter's hand.
    """
    lineups_long = pd.read_parquet(lineups_root / f"lineups_long_{year}.parquet")
    feat = pd.read_parquet(features_path,
                           columns=["game_id", "game_date", "home_id", "away_id",
                                     "home_starter_id", "away_starter_id"])
    feat["game_date"] = pd.to_datetime(feat["game_date"])

    # Build the (team_id, opposing_sp_id) annotation for each game-side.
    home = feat[["game_id", "game_date", "home_id", "away_id",
                 "home_starter_id", "away_starter_id"]].rename(
        columns={"home_id": "team_id", "away_id": "opp_team_id",
                 "away_starter_id": "opp_sp_id", "home_starter_id": "team_sp_id"})
    home["side"] = "home"
    away = feat[["game_id", "game_date", "home_id", "away_id",
                 "home_starter_id", "away_starter_id"]].rename(
        columns={"away_id": "team_id", "home_id": "opp_team_id",
                 "home_starter_id": "opp_sp_id", "away_starter_id": "team_sp_id"})
    away["side"] = "away"
    annot = pd.concat([home, away], ignore_index=True)[
        ["game_id", "game_date", "team_id", "opp_sp_id", "side"]
    ]

    long = lineups_long.merge(annot, on=["game_id", "side"], how="inner")
    return long


def _add_opposing_hand(long_df: pd.DataFrame,
                       hand_map: dict[int, str]) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["opp_sp_hand"] = long_df["opp_sp_id"].map(
        lambda pid: hand_map.get(int(pid)) if pd.notna(pid) else None
    )
    return long_df


# ---------------------------------------------------------------------------
# Slot-level mode with recency tiebreak
# ---------------------------------------------------------------------------

def _resolve_duplicate_positions(
    player_ids: list[int],
    positions: list[str | None],
    games_by_pp: dict[tuple[int, str], int],
) -> tuple[list[str | None], list[tuple[int, str, str]]]:
    """Demote second-and-subsequent players at a duplicate fielding position
    to DH, picking the *defensive regular* by games-started at the position.

    For each fielding position (1B/2B/3B/SS/LF/CF/RF) that appears more
    than once in a projected lineup:

      1. Rank the players sharing it by games started at THAT position
         this season (from ``games_by_pp``), descending. The player with
         more starts is the regular fielder.
      2. Keep the top-ranked player at the fielding position.
      3. The other player(s) get their position changed to ``'DH'`` —
         they remain in the offensive lineup but contribute zero to OAA
         / defense aggregation.

    Ties on games are broken by batting-order index (lower slot = more
    established starter, kept at the fielding position).

    Catcher dupes are left untouched (both already in SKIP_POSITIONS so
    defense correctly contributes zero either way).

    Returns ``(new_positions, demotions)`` where ``demotions`` is a list
    of ``(player_id, original_position, new_position)`` for logging /
    debugging.
    """
    new_positions = list(positions)
    by_pos: dict[str, list[int]] = {}
    for i, pos in enumerate(new_positions):
        if pos in FIELDING_POSITIONS:
            by_pos.setdefault(pos, []).append(i)
    demotions: list[tuple[int, str, str]] = []
    for pos, slot_indices in by_pos.items():
        if len(slot_indices) <= 1:
            continue
        def rank_key(idx: int) -> tuple[int, int]:
            pid = player_ids[idx]
            g = games_by_pp.get((int(pid), pos), 0)
            # Higher games first; tiebreak by lower batting slot.
            return (-g, idx)
        slot_indices.sort(key=rank_key)
        for idx in slot_indices[1:]:
            demotions.append((int(player_ids[idx]), pos, "DH"))
            new_positions[idx] = "DH"
    return new_positions, demotions


def _pick_slot_mode(slot_rows: pd.DataFrame,
                    exclude: set[int]) -> tuple[int | None, str | None]:
    """Return ``(player_id, position)`` for the modal player at this slot,
    skipping any IDs already in ``exclude`` (the slots filled earlier).

    Ties broken by most recent appearance. Returns (None, None) if no
    eligible player has any appearances at this slot.
    """
    if slot_rows.empty:
        return None, None
    eligible = slot_rows[~slot_rows["player_id"].isin(exclude)]
    if eligible.empty:
        return None, None
    last_appearance = (
        eligible.sort_values("game_date")
        .groupby("player_id", as_index=False)
        .agg(n=("player_id", "size"),
             last_date=("game_date", "max"))
    )
    if last_appearance.empty:
        return None, None
    best = last_appearance.sort_values(["n", "last_date"], ascending=[False, False]).iloc[0]
    pid = int(best["player_id"])
    pos_series = eligible.loc[eligible["player_id"] == pid, "position"].dropna()
    pos: str | None = None
    if not pos_series.empty:
        pos = pos_series.mode().iloc[0]
    return pid, pos


def _pick_modal_at_position(pool: pd.DataFrame, pos: str,
                             exclude: set[int]) -> int | None:
    """Pick the player with the most pool starts at ``pos`` (excluding
    already-assigned IDs). Ties broken by most recent appearance.

    Kept for backwards compatibility / single-position debugging. The
    main projector now uses :func:`_assign_lineup_hungarian` which
    solves the entire lineup jointly.
    """
    rows = pool[(pool["position"] == pos)
                & (~pool["player_id"].isin(exclude))
                & pool["player_id"].notna()]
    if rows.empty:
        return None
    agg = (rows.sort_values("game_date")
           .groupby("player_id", as_index=False)
           .agg(n=("player_id", "size"), last_date=("game_date", "max")))
    if agg.empty:
        return None
    best = agg.sort_values(["n", "last_date"], ascending=[False, False]).iloc[0]
    return int(best["player_id"])


# How many top-by-total-starts players to consider as candidates for the
# Hungarian assignment. We have 9 positions to fill; 15 gives the
# solver enough slack to find good rotation-regular fits without
# letting low-PA call-ups crowd in.
HUNGARIAN_TOP_N: int = 15


def _assign_lineup_hungarian(
    pool: pd.DataFrame,
    *,
    positions: tuple[str, ...] = POSITION_PRIORITY,
    top_n: int = HUNGARIAN_TOP_N,
) -> list[tuple[int, str]]:
    """Hungarian (max-weight bipartite matching) lineup assignment.

    1. Rank players in ``pool`` by total starts (any position),
       descending. Keep the top ``top_n``.
    2. Build a cost matrix ``[top_n × len(positions)]`` where cell
       ``[i, j]`` = number of pool starts player ``i`` has at
       position ``j``.
    3. Run ``scipy.optimize.linear_sum_assignment(maximize=True)`` to
       find the assignment that maximizes total weight. The solver
       returns one player per position.

    Returns ``[(player_id, position), ...]`` sorted by position
    priority. Empty list if the pool is empty.

    Edge cases:
      * If a player has zero starts at any position in ``positions``
        (e.g. only ever DH'd elsewhere), they may still be assigned
        somewhere if there aren't enough alternatives — the solver
        will pick a zero-weight pair rather than leave a position
        empty. This is correct behavior for a "best 9 we can field"
        lineup; downstream feature compute will use this player's
        statcast splits regardless of the projected position.
      * If the pool has fewer than 9 unique players, the cost matrix
        is padded with zero rows so the solver still runs to
        completion. The "phantom" player slots map to ``(0, pos)``.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    if pool.empty:
        return []
    pool = pool[pool["player_id"].notna()]
    total = pool.groupby("player_id").size().sort_values(ascending=False)
    candidate_pids: list[int] = [int(p) for p in total.head(top_n).index]
    if not candidate_pids:
        return []

    pos_list = list(positions)
    counts = (pool[pool["player_id"].isin(candidate_pids)]
              .groupby(["player_id", "position"]).size()
              .unstack(fill_value=0))
    counts = counts.reindex(index=candidate_pids,
                             columns=pos_list, fill_value=0)
    cost = counts.values.astype(float)

    # Pad rows with zeros if we have fewer candidates than positions.
    n_rows, n_cols = cost.shape
    if n_rows < n_cols:
        pad = np.zeros((n_cols - n_rows, n_cols))
        cost = np.vstack([cost, pad])

    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    pos_index = {p: i for i, p in enumerate(pos_list)}
    assigned: list[tuple[int, str]] = []
    for r, c in zip(row_ind, col_ind):
        pos = pos_list[c]
        if r >= len(candidate_pids):
            assigned.append((0, pos))  # padding row
            continue
        pid = candidate_pids[r]
        assigned.append((pid, pos))
    assigned.sort(key=lambda pair: pos_index[pair[1]])
    return assigned


def _player_mean_slot(pool: pd.DataFrame, pid: int) -> float:
    """Average batting slot for a player in the pool. Used to rank
    chosen players into a batting order. Falls back to 9.0 (bottom of
    order) if the player has no rows.
    """
    rows = pool.loc[pool["player_id"] == pid, "slot"]
    rows = pd.to_numeric(rows, errors="coerce").dropna()
    if rows.empty:
        return 9.0
    return float(rows.mean())


def _project_one_team(team_id: int,
                      hand: str | None,
                      pool_full: pd.DataFrame,
                      as_of_date: date,
                      *,
                      active_roster: set[int] | None = None,
                      games_by_pp: dict[tuple[int, str], int] | None = None,
                      ) -> ProjectedLineup:
    """Position-first projection for one team.

    Algorithm:
      1. Build the pool (same-hand 30d if available, else all-platoon
         14d), IL-filtered.
      2. Iterate ``POSITION_PRIORITY`` and pick the modal-frequency
         player at each position who hasn't been assigned yet.
      3. If we end with <9 players, top up from the season-wide pool
         (still IL-filtered) by picking the most-frequent unassigned
         players and assigning them to the missing positions if they
         have any starts there.
      4. Rank the chosen 9 players by mean batting slot in the pool to
         produce the final 1–9 batting card.

    The legacy ``games_by_pp`` arg is no longer needed for dedup (each
    position is uniquely assigned by construction) but kept on the
    signature for backwards compatibility — it's ignored.
    """
    del games_by_pp  # unused under the new algorithm

    as_of_ts = pd.Timestamp(as_of_date)
    short_cutoff = as_of_ts - pd.Timedelta(days=LOOKBACK_DAYS)
    wide_cutoff = as_of_ts - pd.Timedelta(days=PLATOON_LOOKBACK_DAYS)
    team_rows = pool_full[(pool_full["team_id"] == team_id)
                          & (pool_full["game_date"] < as_of_ts)]
    all_platoon = team_rows[team_rows["game_date"] >= short_cutoff]
    platoon_wide = team_rows[team_rows["game_date"] >= wide_cutoff]

    used_platoon = False
    if hand is not None:
        same_hand_wide = platoon_wide[platoon_wide["opp_sp_hand"] == hand]
        if same_hand_wide["game_id"].nunique() >= MIN_POOL_GAMES:
            pool = same_hand_wide
            used_platoon = True
        else:
            pool = all_platoon
    else:
        pool = all_platoon

    # IL / 40-man / optioned filter — drop anyone not on the active roster.
    if active_roster:
        before = pool["player_id"].nunique()
        pool = pool[pool["player_id"].isin(active_roster)]
        dropped = before - pool["player_id"].nunique()
        if dropped:
            logger.debug("[il-filter] team %s: dropped %d players from pool "
                         "(not on active roster)", team_id, dropped)

    pool_size = pool["game_id"].nunique()

    # --- Hungarian lineup assignment --------------------------------------
    # Jointly pick the 9-player x 9-position assignment that maximizes
    # total games-played in the pool. Captures rotation regulars and
    # conditional position swaps that a greedy per-position pick misses.
    assigned = _assign_lineup_hungarian(pool)

    # --- Top-up if the recent pool had no usable players ------------------
    # Pool is empty (early-season team with no recent games) -> fall back
    # to the season-wide team pool.
    if not assigned:
        season_pool = pool_full[(pool_full["team_id"] == team_id)
                                 & (pool_full["game_date"] < as_of_ts)]
        if active_roster:
            season_pool = season_pool[season_pool["player_id"].isin(active_roster)]
        assigned = _assign_lineup_hungarian(season_pool)

    # --- Order the chosen players into a batting card ---------------------
    # Mean slot from the recent pool (preferred) or season pool (fallback).
    fallback_pool = pool_full[(pool_full["team_id"] == team_id)
                               & (pool_full["game_date"] < as_of_ts)]
    def slot_score(pid: int) -> float:
        s = _player_mean_slot(pool, pid)
        if s == 9.0:
            # Player never appeared in the recent pool — fall back to season.
            s = _player_mean_slot(fallback_pool, pid)
        return s

    assigned.sort(key=lambda pair: slot_score(pair[0]))

    # Pad to 9 slots if (very rarely) we still came up short.
    while len(assigned) < LINEUP_SLOTS:
        assigned.append((0, None))

    player_ids = [pid for pid, _ in assigned]
    positions  = [pos for _, pos in assigned]

    return ProjectedLineup(
        team_id=team_id,
        as_of_date=as_of_date,
        opposing_hand=hand,
        pool_size=pool_size,
        used_platoon_split=used_platoon,
        player_ids=player_ids,
        positions=positions,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def project_lineups_for_schedule(
    schedule_df: pd.DataFrame,
    *,
    year: int,
    lineups_root: Path,
    features_path: Path,
    pitcher_hand_map: dict[int, str] | None = None,
    filter_active_roster: bool = True,
    active_roster_map: dict[int, set[int]] | None = None,
    games_by_pp: dict[tuple[int, str], int] | None = None,
) -> dict[int, dict[str, ProjectedLineup]]:
    """Project both teams' lineups for every game in ``schedule_df``.

    Returns a dict keyed by ``game_id`` -> ``{"home": ProjectedLineup,
    "away": ProjectedLineup}``.

    ``schedule_df`` is what ``fetch_schedule.load_schedule_for_date``
    returns (must have ``game_id``, ``game_date``, ``home_id``,
    ``away_id``, and the two ``*_probable_pitcher_throws`` columns).

    If ``pitcher_hand_map`` is passed, it's used to fill any missing
    ``*_probable_pitcher_throws`` values.
    """
    if schedule_df.empty:
        return {}
    long_df = _load_history(year=year, lineups_root=lineups_root,
                             features_path=features_path)
    # Build the games-started-at-position lookup for duplicate-position
    # resolution. Counted across the WHOLE season so we have a stable
    # signal for who the regular fielder is at each spot.
    if games_by_pp is None:
        from src.features.team_defense import count_games_by_position
        lineups_long_raw = pd.read_parquet(
            lineups_root / f"lineups_long_{year}.parquet"
        )
        games_by_pp = count_games_by_position(lineups_long_raw)
    # Build hand map from training_path (already has all opp SPs).
    if pitcher_hand_map is None:
        pitcher_hand_map = {}
    # If schedule rows have ``*_probable_pitcher_throws`` set, use them as
    # the source of truth; else look up in hand_map; else None.
    long_df["opp_sp_hand"] = None  # filled per-call below
    out: dict[int, dict[str, ProjectedLineup]] = {}

    # We need opp_sp_hand on the long history once, so derive it from
    # both the explicit schedule (which has hand) AND the hand_map.
    # Easiest: compute a per-(opp_sp_id) hand using the schedule for
    # tonight's pitchers and the statcast lookup for everyone else.
    hand_overrides: dict[int, str] = {}
    for _, r in schedule_df.iterrows():
        for side in ("home", "away"):
            pid = r.get(f"{side}_probable_pitcher_id")
            hnd = r.get(f"{side}_probable_pitcher_throws")
            if pd.notna(pid) and pd.notna(hnd):
                hand_overrides[int(pid)] = str(hnd)
    full_hand_map = {**pitcher_hand_map, **hand_overrides}
    long_df = _add_opposing_hand(long_df, full_hand_map)

    # Build active-roster cache (one HTTP call per team in the slate).
    if filter_active_roster and active_roster_map is None:
        team_ids = (schedule_df["home_id"].tolist()
                    + schedule_df["away_id"].tolist())
        active_roster_map = build_active_roster_map(team_ids)
    elif not filter_active_roster:
        active_roster_map = {}

    for _, r in schedule_df.iterrows():
        gid = int(r["game_id"])
        as_of: date = (r["game_date"].date()
                       if hasattr(r["game_date"], "date")
                       else date.fromisoformat(str(r["game_date"])[:10]))
        away_pp_hand = r.get("away_probable_pitcher_throws")
        home_pp_hand = r.get("home_probable_pitcher_throws")
        home = _project_one_team(
            team_id=int(r["home_id"]),
            hand=str(away_pp_hand) if pd.notna(away_pp_hand) else None,
            pool_full=long_df,
            as_of_date=as_of,
            active_roster=active_roster_map.get(int(r["home_id"])),
            games_by_pp=games_by_pp,
        )
        away = _project_one_team(
            team_id=int(r["away_id"]),
            hand=str(home_pp_hand) if pd.notna(home_pp_hand) else None,
            pool_full=long_df,
            as_of_date=as_of,
            active_roster=active_roster_map.get(int(r["away_id"])),
            games_by_pp=games_by_pp,
        )
        out[gid] = {"home": home, "away": away}
    return out


def project_lineup(
    team: int,
    as_of_date: str | date,
    opposing_pitcher_hand: str | None,
    *,
    year: int,
    lineups_root: Path = Path("data/lineups"),
    features_path: Path | None = None,
    filter_active_roster: bool = True,
) -> ProjectedLineup:
    """Single-team convenience wrapper. Mostly useful for ad-hoc debugging
    from a notebook.
    """
    from src.ingest.fetch_schedule import pitcher_hand_lookup

    if isinstance(as_of_date, str):
        as_of_date = date.fromisoformat(as_of_date)
    if features_path is None:
        features_path = Path(f"data/features/training_{year}.parquet")
    from src.features.team_defense import count_games_by_position

    long_df = _load_history(year=year, lineups_root=lineups_root,
                             features_path=features_path)
    hand_map = pitcher_hand_lookup(years=(year,))
    long_df = _add_opposing_hand(long_df, hand_map)
    active = fetch_active_roster_ids(int(team)) if filter_active_roster else None
    lineups_long_raw = pd.read_parquet(lineups_root / f"lineups_long_{year}.parquet")
    games_by_pp = count_games_by_position(lineups_long_raw)
    return _project_one_team(team_id=int(team), hand=opposing_pitcher_hand,
                              pool_full=long_df, as_of_date=as_of_date,
                              active_roster=active,
                              games_by_pp=games_by_pp)
