"""Per-game bullpen feature engineering.

The starter currently carries 100% of the model's pitching signal, but
modern SPs only throw ~5.0-5.5 IP per game — meaning we're implicitly
treating the other ~3.5 IP as league-average. This module fixes that
by building per-team, per-handedness bullpen composites that get
attached to the per-game training frame alongside the starter features.

Design (per the spec locked in with the user):

* **Composition** — for every (team, date), take the top 8 relievers
  by last-30-day IP. Naturally excludes IL'd, DFA'd, and minor-league
  pitchers because they have no recent appearances.
* **Weighting** — IP-weighted average per handedness pool. Heavier-used
  pitchers (high-leverage closers, workhorse setup men) count more.
* **Rest rule** — a pitcher who appeared yesterday AND threw 20+
  pitches is excluded from today's pool (likely unavailable). One-batter
  cameos don't trigger the exclusion. B2B-day closers who throw <20
  pitches still count.
* **Handedness** — pool is split into R and L subpools. Aggregates
  emitted as ``bp_R_*`` and ``bp_L_*`` so the model can platoon-weight.
* **Matchup adjustment** — applied identically to how it's done for
  starters (per-flight park × per-handedness OAA), using the pool's
  IP-weighted flight rates as the "team-BP-as-one-pitcher" profile.

Top-level entry points:

* :func:`build_appearances` — per-(pitcher, game) row table, the
  ground truth for IP / pitches / role / team / handedness.
* :func:`compute_reliever_stats` — relief-pitches-only cumulative and
  rolling-30d aggregates, same schema as
  :func:`src.features.cumulative.build_cumulative` outputs.
* :func:`build_pool_lookup` — ``(team_id, as_of_date) -> [(pid, weight), ...]``
  applying the top-N + rest rule.
* :func:`attach_bullpen_composites` — attaches the ``<side>_bp_{R,L}_*``
  columns (with matchup_adj) to a games DataFrame.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.features.cumulative import (
    build_cumulative, build_rolling, compute_rate_stats,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pool composition / rest rule (locked in with the user).
BULLPEN_POOL_TOP_N: int = 8
BULLPEN_RECENT_WINDOW_DAYS: int = 30
BULLPEN_REST_PITCH_THRESHOLD: int = 20

# Out counts per PA-ending event. We use these to estimate IP per
# (pitcher, game) from raw Statcast. Approximate — ignores base-running
# outs (caught stealings, pickoffs) which aren't PA-ending. Close enough
# for IP-weighting purposes; we're not trying to match the box-score IP
# to the third.
_OUTS_PER_EVENT: dict[str, int] = {
    "field_out": 1, "force_out": 1, "fielders_choice_out": 1,
    "strikeout": 1, "sac_fly": 1, "sac_bunt": 1,
    "grounded_into_double_play": 2, "double_play": 2,
    "strikeout_double_play": 2, "sac_fly_double_play": 2,
    "triple_play": 3,
}

# Feature columns we surface per reliever from the cumulative aggregator.
RELIEVER_FEATURE_COLS: tuple[str, ...] = (
    "xwOBA", "SIERA", "Barrel_pct",
    "GB_pct", "FB_pct", "LD_pct", "PU_pct",
    "PA_cum",
)


# ---------------------------------------------------------------------------
# Per-(pitcher, game) appearance table
# ---------------------------------------------------------------------------

def _outs_for_events(events: pd.Series) -> pd.Series:
    """Vectorized event -> outs lookup. Returns int Series same length."""
    return events.map(_OUTS_PER_EVENT).fillna(0).astype(int)


def _team_lookup_from_boxscores(boxscores_wide: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, pitcher_id, team_id, is_starter).

    The boxscore's ``home_pitchers_used`` / ``away_pitchers_used`` are
    lists of every pitcher who appeared, in order — the first is the
    starter. We explode both sides into a long table that downstream
    code joins onto Statcast pitches.
    """
    pieces = []
    for side in ("home", "away"):
        used_col = f"{side}_pitchers_used"
        team_col = f"{side}_team_id"
        starter_col = f"{side}_starter_id"
        sub = boxscores_wide[["game_id", team_col, used_col, starter_col]].copy()
        sub = sub.explode(used_col, ignore_index=True)
        sub = sub.dropna(subset=[used_col])
        sub["pitcher_id"] = pd.to_numeric(sub[used_col], errors="coerce").astype("Int64")
        sub["team_id"] = pd.to_numeric(sub[team_col], errors="coerce").astype("Int64")
        sub["is_starter"] = sub["pitcher_id"] == pd.to_numeric(
            sub[starter_col], errors="coerce"
        ).astype("Int64")
        sub["side"] = side
        pieces.append(sub[["game_id", "pitcher_id", "team_id", "side", "is_starter"]])
    out = pd.concat(pieces, ignore_index=True).dropna(subset=["pitcher_id"])
    return out.drop_duplicates(["game_id", "pitcher_id"]).reset_index(drop=True)


def build_appearances(
    statcast: pd.DataFrame,
    boxscores_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per (pitcher, game) appearance.

    Columns:
        ``game_id``, ``game_date``, ``pitcher_id``, ``team_id``, ``side``
        (home/away), ``is_starter`` (bool), ``p_throws`` (R/L),
        ``pitches`` (count of pitches thrown), ``outs`` (sum of outs
        recorded via PA-ending events), ``IP`` (= outs / 3.0).

    This is the foundation for everything else: the pool lookup uses
    ``IP`` to rank and weight relievers, and the rest rule uses
    ``pitches`` on the day-before to flag unavailability.
    """
    sc = statcast.copy()
    sc["game_id"] = pd.to_numeric(sc["game_pk"], errors="coerce").astype("Int64")
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    sc["pitcher_id"] = pd.to_numeric(sc["pitcher"], errors="coerce").astype("Int64")

    pitches = (
        sc.groupby(["game_id", "pitcher_id", "game_date"], as_index=False)
        .agg(pitches=("pitcher_id", "size"))
    )

    pa = sc[sc["events"].notna()].copy()
    pa["outs"] = _outs_for_events(pa["events"])
    outs = (
        pa.groupby(["game_id", "pitcher_id"], as_index=False)
        .agg(outs=("outs", "sum"))
    )

    handedness = (
        sc.dropna(subset=["p_throws"])
        .groupby("pitcher_id")["p_throws"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("p_throws")
        .reset_index()
    )

    team_lookup = _team_lookup_from_boxscores(boxscores_wide)

    out = (
        pitches.merge(outs, on=["game_id", "pitcher_id"], how="left")
        .merge(team_lookup, on=["game_id", "pitcher_id"], how="left")
        .merge(handedness, on="pitcher_id", how="left")
    )
    out["outs"] = out["outs"].fillna(0).astype(int)
    out["IP"] = out["outs"] / 3.0
    out["pitches"] = out["pitches"].astype(int)
    return out.dropna(subset=["team_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-reliever cumulative / rolling stats
# ---------------------------------------------------------------------------

def filter_relief_pitches(
    statcast: pd.DataFrame,
    appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Return only the Statcast rows where the pitcher was a RELIEVER
    in that game (not the starter of record).

    Same pitcher can be a starter in one game (his scheduled start) and
    a reliever in another (rare emergency / opener-then-bulk situations).
    We only keep the relief appearances for the per-reliever aggregates,
    so a swingman's bulk-innings show up in SP stats and his short relief
    outings show up in BP stats — neither bleeds into the other.
    """
    relief_keys = appearances.loc[
        ~appearances["is_starter"], ["game_id", "pitcher_id"]
    ].copy()
    relief_keys["pitcher"] = relief_keys["pitcher_id"].astype("int64")
    relief_keys["game_pk"] = relief_keys["game_id"].astype("int64")
    sc = statcast.copy()
    sc["pitcher"] = pd.to_numeric(sc["pitcher"], errors="coerce").astype("Int64")
    sc["game_pk"] = pd.to_numeric(sc["game_pk"], errors="coerce").astype("Int64")
    keep = relief_keys[["game_pk", "pitcher"]].drop_duplicates()
    return sc.merge(keep, on=["game_pk", "pitcher"], how="inner")


def compute_reliever_stats(
    statcast: pd.DataFrame,
    appearances: pd.DataFrame,
    *,
    rolling_window_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(cumulative, rolling)`` reliever rate stats.

    Re-uses the existing ``build_cumulative`` / ``build_rolling`` /
    ``compute_rate_stats`` pipeline on the relief-only Statcast subset,
    so the schema matches what SP features already use (xwOBA, SIERA,
    Barrel%, flight rates, PA_cum). Downstream consumers can treat
    these tables identically to ``cum_p`` and ``rol_p``.
    """
    relief_sc = filter_relief_pitches(statcast, appearances)
    if relief_sc.empty:
        empty = pd.DataFrame(columns=["player_id", "game_year", "game_date",
                                       *RELIEVER_FEATURE_COLS])
        return empty, empty
    cum = compute_rate_stats(build_cumulative(relief_sc, group="pitcher"))
    rol = compute_rate_stats(build_rolling(
        relief_sc, group="pitcher", window_days=rolling_window_days,
    ))
    return cum, rol


# ---------------------------------------------------------------------------
# Pool lookup: (team, date) -> [(pitcher_id, weight)]
# ---------------------------------------------------------------------------

def _previous_day_pitch_lookup(
    appearances: pd.DataFrame,
) -> dict[tuple[int, pd.Timestamp], int]:
    """``(pitcher_id, game_date) -> pitches thrown that day``.

    Used to apply the rest rule: if a pitcher threw ≥ 20 pitches the
    DAY BEFORE the target date, they're excluded.
    """
    daily = (
        appearances.groupby(["pitcher_id", "game_date"], as_index=False)
        .agg(pitches=("pitches", "sum"))
    )
    return {
        (int(pid), pd.Timestamp(d)): int(p)
        for pid, d, p in zip(daily["pitcher_id"], daily["game_date"], daily["pitches"])
    }


def build_pool_lookup(
    appearances: pd.DataFrame,
    *,
    top_n: int = BULLPEN_POOL_TOP_N,
    recent_days: int = BULLPEN_RECENT_WINDOW_DAYS,
    rest_pitch_threshold: int = BULLPEN_REST_PITCH_THRESHOLD,
    extra_team_dates: list[tuple[int, pd.Timestamp]] | None = None,
) -> dict[tuple[int, pd.Timestamp], list[tuple[int, float]]]:
    """Build ``(team_id, as_of_date) -> [(pitcher_id, last_30d_IP), ...]``.

    The list contains at most ``top_n`` entries, sorted by descending
    IP, after applying the rest rule (drop pitchers who threw
    ``rest_pitch_threshold``+ pitches the day before).

    "As-of date" means: relief appearances that occurred strictly BEFORE
    that date. No lookahead.

    Performance note: this is computed once per (team, date) pair we'll
    ever query — i.e. one entry per team per game-day in the season. For
    a 162-game season with 30 teams that's ~5,000 keys; quite manageable.
    """
    relief = appearances.loc[~appearances["is_starter"]].copy()
    relief["team_id"] = relief["team_id"].astype("int64")
    relief["pitcher_id"] = relief["pitcher_id"].astype("int64")
    relief["game_date"] = pd.to_datetime(relief["game_date"])

    yesterday_pitches = _previous_day_pitch_lookup(appearances)

    # Build per-team rolling pool. The boundary dates we care about are
    # the game_dates per team — relief appearances on those dates plus
    # one day in either direction.
    pool: dict[tuple[int, pd.Timestamp], list[tuple[int, float]]] = {}

    # Pre-sort once for the rolling window scan.
    relief = relief.sort_values(["team_id", "game_date"])

    # Build the per-team "dates to compute" set. By default this is
    # every date a team has a relief appearance. ``extra_team_dates``
    # lets a caller request pools for additional dates (e.g. tonight's
    # projected slate — today won't be in the historical appearances).
    extras_by_team: dict[int, set[pd.Timestamp]] = {}
    if extra_team_dates:
        for t, d in extra_team_dates:
            extras_by_team.setdefault(int(t), set()).add(pd.Timestamp(d))

    # For each team, walk the unique team-dates we need pools for.
    # Compute pools for every date a team appears in `relief` plus any
    # extras the caller requested. Callers query by exact (team_id, date);
    # missing keys -> empty pool.
    team_ids = set(relief["team_id"].astype("int64").unique()) | set(extras_by_team.keys())
    for team_id in sorted(team_ids):
        sub = relief.loc[relief["team_id"] == team_id].sort_values("game_date").reset_index(drop=True)
        historical_dates = set(pd.unique(sub["game_date"])) if not sub.empty else set()
        team_dates = sorted(historical_dates | extras_by_team.get(int(team_id), set()))
        for as_of in team_dates:
            target = pd.Timestamp(as_of)
            # Recent window: [as_of - recent_days, as_of) (strict <)
            lo = target - pd.Timedelta(days=recent_days)
            recent = sub[(sub["game_date"] >= lo) & (sub["game_date"] < target)]
            if recent.empty:
                pool[(int(team_id), target)] = []
                continue
            ip_by_pid = recent.groupby("pitcher_id")["IP"].sum().sort_values(
                ascending=False
            )
            # Rest rule: drop pitchers who threw >= threshold on (target - 1d).
            yesterday = target - pd.Timedelta(days=1)
            keep: list[tuple[int, float]] = []
            for pid, ip in ip_by_pid.items():
                pitches_yest = yesterday_pitches.get((int(pid), yesterday), 0)
                if pitches_yest >= rest_pitch_threshold:
                    continue
                keep.append((int(pid), float(ip)))
                if len(keep) >= top_n:
                    break
            pool[(int(team_id), target)] = keep
    return pool


# ---------------------------------------------------------------------------
# Team-level composites (per handedness)
# ---------------------------------------------------------------------------

def compute_pool_composites(
    pool_lookup: Mapping[tuple[int, pd.Timestamp], list[tuple[int, float]]],
    reliever_stats: pd.DataFrame,
    pitcher_handedness: Mapping[int, str],
    *,
    feature_cols: Iterable[str] = RELIEVER_FEATURE_COLS,
) -> pd.DataFrame:
    """Per ``(team_id, as_of_date)``: IP-weighted average of each feature
    in ``feature_cols``, split by reliever handedness.

    Output columns:
        ``team_id``, ``game_date``, ``pool_size``,
        ``R_pool_size``, ``L_pool_size``,
        ``R_<feature>``, ``L_<feature>`` for each feature in feature_cols.

    Missing per-handedness pools (e.g. no LHPs in the top 8) emit NaN
    rather than 0 — downstream Ridge/imputation handles them gracefully.
    """
    rs = reliever_stats.copy()
    rs["player_id"] = pd.to_numeric(rs["player_id"], errors="coerce").astype("int64")
    rs["game_date"] = pd.to_datetime(rs["game_date"])
    rs = rs.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    feature_cols = list(feature_cols)

    # Per-pitcher: store the date-sorted (date, feature_vector) for fast as-of lookup.
    per_pid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, sub in rs.groupby("player_id", sort=False):
        dates = sub["game_date"].to_numpy()
        feats = sub[feature_cols].to_numpy(dtype=float)
        per_pid[int(pid)] = (dates, feats)

    def _features_asof(pid: int, as_of: pd.Timestamp) -> np.ndarray | None:
        if pid not in per_pid:
            return None
        dates, feats = per_pid[pid]
        idx = np.searchsorted(dates, np.datetime64(as_of), side="left") - 1
        if idx < 0:
            return None
        return feats[idx]

    rows = []
    for (team_id, as_of), members in pool_lookup.items():
        R_pids: list[tuple[int, float]] = []
        L_pids: list[tuple[int, float]] = []
        for pid, weight in members:
            hand = pitcher_handedness.get(int(pid))
            if hand == "R":
                R_pids.append((pid, weight))
            elif hand == "L":
                L_pids.append((pid, weight))
            # Unknown handedness: drop from per-hand pool. Rare.

        rec: dict = {
            "team_id": int(team_id),
            "game_date": pd.Timestamp(as_of),
            "pool_size": len(members),
            "R_pool_size": len(R_pids),
            "L_pool_size": len(L_pids),
        }
        for hand, pids in (("R", R_pids), ("L", L_pids)):
            if not pids:
                for c in feature_cols:
                    rec[f"{hand}_{c}"] = np.nan
                continue
            num = np.zeros(len(feature_cols), dtype=float)
            den = np.zeros(len(feature_cols), dtype=float)
            for pid, weight in pids:
                vec = _features_asof(pid, as_of)
                if vec is None:
                    continue
                # IP-weight each feature; per-feature `den` so a single
                # NaN doesn't kill the whole pool's average.
                w = float(weight)
                valid = ~np.isnan(vec)
                num[valid] += vec[valid] * w
                den[valid] += w
            avg = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
            for c, v in zip(feature_cols, avg):
                rec[f"{hand}_{c}"] = float(v) if not np.isnan(v) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level convenience: build everything needed for build_features.py
# ---------------------------------------------------------------------------

def build_bullpen_features(
    statcast: pd.DataFrame,
    boxscores_wide: pd.DataFrame,
    *,
    rolling_window_days: int = 30,
    top_n: int = BULLPEN_POOL_TOP_N,
    recent_days: int = BULLPEN_RECENT_WINDOW_DAYS,
    rest_pitch_threshold: int = BULLPEN_REST_PITCH_THRESHOLD,
    extra_team_dates: list[tuple[int, pd.Timestamp]] | None = None,
) -> dict:
    """One-shot orchestration: appearances + reliever stats (cum + rolling)
    + pool lookup + per-handedness composites (cum + rolling).

    Returns a dict with keys:
        * ``appearances`` — DataFrame
        * ``cum_stats`` / ``rol_stats`` — per-reliever DataFrames
        * ``pool_lookup`` — (team_id, game_date) -> [(pid, IP), ...]
        * ``cum_composites`` / ``rol_composites`` — (team_id, game_date) wide
          frames with R/L pool sizes and per-handedness feature averages
        * ``pitcher_handedness`` — {pitcher_id: 'R' | 'L'} cached lookup
    """
    appearances = build_appearances(statcast, boxscores_wide)
    cum_stats, rol_stats = compute_reliever_stats(
        statcast, appearances, rolling_window_days=rolling_window_days,
    )
    pool_lookup = build_pool_lookup(
        appearances,
        top_n=top_n, recent_days=recent_days,
        rest_pitch_threshold=rest_pitch_threshold,
        extra_team_dates=extra_team_dates,
    )
    pitcher_handedness = dict(
        zip(
            appearances["pitcher_id"].astype(int),
            appearances["p_throws"].astype(str),
        )
    )
    cum_comp = compute_pool_composites(
        pool_lookup, cum_stats, pitcher_handedness,
    )
    rol_comp = compute_pool_composites(
        pool_lookup, rol_stats, pitcher_handedness,
    )
    return {
        "appearances": appearances,
        "cum_stats": cum_stats,
        "rol_stats": rol_stats,
        "pool_lookup": pool_lookup,
        "cum_composites": cum_comp,
        "rol_composites": rol_comp,
        "pitcher_handedness": pitcher_handedness,
    }
