"""Point-in-time team rolling runs baselines (park-adjusted, shrunk).

Builds home/away 30-day rolling runs scored (park-neutralized) and shrinks
toward league average until a team has ``min_games_for_full_weight`` games
in the rolling window (default 30).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.handedness import MLBAM_TEAM_ID_TO_ABBREV, build_park_factor_lookup

DEFAULT_PARK_FACTORS_PATH = Path("data/park_factors/park_factors_2024_rolling3.parquet")
ROLLING_WINDOW_DAYS = 30
MIN_GAMES_FULL_WEIGHT = 30
LEAGUE_AVG_RUNS_FALLBACK = 4.5


@dataclass(frozen=True)
class TeamBaselineConfig:
    rolling_days: int = ROLLING_WINDOW_DAYS
    min_games_full_weight: int = MIN_GAMES_FULL_WEIGHT
    park_factors_path: Path = DEFAULT_PARK_FACTORS_PATH


def _venue_park_runs_factors(park_path: Path) -> dict[int, float]:
    """``home_team_id`` → neutralized runs park multiplier (mean R/L)."""
    if not park_path.exists():
        return {}
    pf = pd.read_parquet(park_path)
    if "index_runs" not in pf.columns:
        return {}
    lookup = build_park_factor_lookup(pf, factor_col="index_runs")
    by_team: dict[int, list[float]] = {}
    for (abbrev, _stand), val in lookup.items():
        tid = next(
            (k for k, v in MLBAM_TEAM_ID_TO_ABBREV.items() if v == abbrev),
            None,
        )
        if tid is not None:
            by_team.setdefault(tid, []).append(val)
    return {tid: float(np.mean(vals)) for tid, vals in by_team.items()}


def _park_adj_runs(runs: pd.Series, home_id: pd.Series, park: dict[int, float]) -> np.ndarray:
    factors = home_id.map(lambda x: park.get(int(x), 1.0) if pd.notna(x) else 1.0)
    factors = np.maximum(factors.astype(float), 0.75)
    return (runs.astype(float) / factors).to_numpy()


def build_team_game_log(games: pd.DataFrame, *, park: dict[int, float]) -> pd.DataFrame:
    """Long team-game log with park-adjusted runs scored / allowed."""
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["home_id"] = pd.to_numeric(g["home_id"], errors="coerce")
    g["away_id"] = pd.to_numeric(g["away_id"], errors="coerce")
    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    g["away_score"] = pd.to_numeric(g["away_score"], errors="coerce")
    if "season_year" not in g.columns:
        g["season_year"] = g["game_date"].dt.year

    park_adj_home = _park_adj_runs(g["home_score"], g["home_id"], park)
    park_adj_away = _park_adj_runs(g["away_score"], g["home_id"], park)

    home_rows = pd.DataFrame({
        "game_id": g["game_id"],
        "game_date": g["game_date"],
        "season_year": g["season_year"],
        "team_id": g["home_id"],
        "side": "home",
        "rs_raw": g["home_score"].astype(float),
        "rs_park_adj": park_adj_home,
        "ra_park_adj": park_adj_away,
    })
    away_rows = pd.DataFrame({
        "game_id": g["game_id"],
        "game_date": g["game_date"],
        "season_year": g["season_year"],
        "team_id": g["away_id"],
        "side": "away",
        "rs_raw": g["away_score"].astype(float),
        "rs_park_adj": park_adj_away,
        "ra_park_adj": park_adj_home,
    })
    return pd.concat([home_rows, away_rows], ignore_index=True)


def _rolling_prior_stats(
    log: pd.DataFrame,
    *,
    rolling_days: int,
) -> pd.DataFrame:
    """Per team-game: 30d calendar RS mean + season-to-date game counts."""
    log = log.sort_values(["team_id", "side", "game_date", "game_id"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []

    # Prior season games (reset each year) for shrinkage weight + bet eligibility.
    season_counts: list[pd.DataFrame] = []
    for (team_id, season_year), grp in log.groupby(["team_id", "season_year"], sort=False):
        g = grp.sort_values("game_date").drop_duplicates("game_id", keep="first")
        season_n = np.arange(len(g))
        season_counts.append(
            pd.DataFrame({
                "game_id": g["game_id"].to_numpy(),
                "team_id": team_id,
                "season_year": season_year,
                "team_season_games": season_n,
            })
        )
    season_lookup = pd.concat(season_counts, ignore_index=True)

    for (team_id, side), grp in log.groupby(["team_id", "side"], sort=False):
        grp = grp.sort_values("game_date").copy()
        grp = grp.set_index("game_date")
        rs_mean = grp["rs_park_adj"].rolling(f"{rolling_days}D", closed="left").mean()
        rs_raw_mean = grp["rs_raw"].rolling(f"{rolling_days}D", closed="left").mean()
        rs_cal_n = grp["rs_park_adj"].rolling(f"{rolling_days}D", closed="left").count()
        ra_mean = grp["ra_park_adj"].rolling(f"{rolling_days}D", closed="left").mean()
        out = grp.reset_index()
        out["team_id"] = team_id
        out["side"] = side
        out["team_rs_roll"] = rs_mean.to_numpy()
        out["team_rs_raw_roll"] = rs_raw_mean.to_numpy()
        out["team_rs_cal_n"] = rs_cal_n.to_numpy()
        out["team_ra_roll"] = ra_mean.to_numpy()
        out = out.merge(
            season_lookup[season_lookup["team_id"] == team_id][
                ["game_id", "team_id", "team_season_games"]
            ],
            on=["game_id", "team_id"],
            how="left",
        )
        pieces.append(out)

    return pd.concat(pieces, ignore_index=True)


def shrink_baseline(
    team_roll: np.ndarray | pd.Series,
    n_games: np.ndarray | pd.Series,
    league_avg: float,
    *,
    min_games_full_weight: int,
) -> np.ndarray:
    """Blend rolling team RS with league average: w = min(1, n / min_games)."""
    roll = np.asarray(team_roll, dtype=float)
    n = np.asarray(n_games, dtype=float)
    w = np.minimum(1.0, n / float(min_games_full_weight))
    out = np.where(np.isnan(roll), league_avg, w * roll + (1.0 - w) * league_avg)
    return np.where(n < 1, league_avg, out)


def league_avg_park_adj_runs(log: pd.DataFrame) -> float:
    if log.empty:
        return LEAGUE_AVG_RUNS_FALLBACK
    return float(log["rs_park_adj"].mean())


def league_avg_raw_runs(log: pd.DataFrame) -> float:
    if log.empty or "rs_raw" not in log.columns:
        return LEAGUE_AVG_RUNS_FALLBACK
    return float(log["rs_raw"].mean())


def attach_team_offense_baselines(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    *,
    league_avg: float,
    min_games_full_weight: int,
) -> pd.DataFrame:
    """Add per-game home/away shrunk offense baselines + sample counts."""
    out = games.copy()
    home_stats = team_stats[team_stats["side"] == "home"].rename(columns={
        "team_rs_roll": "home_team_rs_roll",
        "team_rs_raw_roll": "home_team_rs_raw_roll",
        "team_rs_cal_n": "home_team_rs_cal_n",
        "team_ra_roll": "home_team_ra_roll",
        "team_season_games": "home_team_season_games",
    })
    away_stats = team_stats[team_stats["side"] == "away"].rename(columns={
        "team_rs_roll": "away_team_rs_roll",
        "team_rs_raw_roll": "away_team_rs_raw_roll",
        "team_rs_cal_n": "away_team_rs_cal_n",
        "team_ra_roll": "away_team_ra_roll",
        "team_season_games": "away_team_season_games",
    })
    out = out.merge(
        home_stats[[
            "game_id", "team_id", "home_team_rs_roll", "home_team_rs_raw_roll",
            "home_team_rs_cal_n", "home_team_ra_roll", "home_team_season_games",
        ]],
        left_on=["game_id", "home_id"],
        right_on=["game_id", "team_id"],
        how="left",
    ).drop(columns=["team_id"], errors="ignore")
    out = out.merge(
        away_stats[[
            "game_id", "team_id", "away_team_rs_roll", "away_team_rs_raw_roll",
            "away_team_rs_cal_n", "away_team_ra_roll", "away_team_season_games",
        ]],
        left_on=["game_id", "away_id"],
        right_on=["game_id", "team_id"],
        how="left",
    ).drop(columns=["team_id"], errors="ignore")

    # Shrink toward league avg by season games played (full team weight at 30 games).
    out["home_off_baseline"] = shrink_baseline(
        out["home_team_rs_roll"], out["home_team_season_games"], league_avg,
        min_games_full_weight=min_games_full_weight,
    )
    out["away_off_baseline"] = shrink_baseline(
        out["away_team_rs_roll"], out["away_team_season_games"], league_avg,
        min_games_full_weight=min_games_full_weight,
    )
    out["prototype_min_team_games"] = np.minimum(
        out["home_team_season_games"].fillna(0),
        out["away_team_season_games"].fillna(0),
    )
    out["prototype_bet_eligible"] = out["prototype_min_team_games"] >= min_games_full_weight
    return out


def build_baselines_for_games(
    history_games: pd.DataFrame,
    target_games: pd.DataFrame,
    *,
    config: TeamBaselineConfig | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Compute rolling team stats from *history* and attach to *target*."""
    cfg = config or TeamBaselineConfig()
    park = _venue_park_runs_factors(cfg.park_factors_path)
    hist_log = build_team_game_log(history_games, park=park)
    league_avg = league_avg_park_adj_runs(hist_log)
    league_avg_raw = league_avg_raw_runs(hist_log)
    team_stats = _rolling_prior_stats(hist_log, rolling_days=cfg.rolling_days)

    # Target games need rows in team_stats too — concat history+target for
    # rolling, but only return target rows' attached baselines.
    if target_games is history_games:
        attached = attach_team_offense_baselines(
            target_games, team_stats, league_avg=league_avg,
            min_games_full_weight=cfg.min_games_full_weight,
        )
        attached["league_avg_raw_runs"] = league_avg_raw
        return attached, league_avg, league_avg_raw

    comb = pd.concat([history_games, target_games], ignore_index=True).drop_duplicates("game_id")
    comb_log = build_team_game_log(comb, park=park)
    comb_stats = _rolling_prior_stats(comb_log, rolling_days=cfg.rolling_days)
    target_ids = set(target_games["game_id"].astype(int))
    comb_stats = comb_stats[comb_stats["game_id"].isin(target_ids)]
    attached = attach_team_offense_baselines(
        target_games, comb_stats, league_avg=league_avg,
        min_games_full_weight=cfg.min_games_full_weight,
    )
    attached["league_avg_raw_runs"] = league_avg_raw
    return attached, league_avg, league_avg_raw


def attach_team_offense_norms(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    *,
    rolling_days: int = ROLLING_WINDOW_DAYS,
    off_col_home: str = "home_off_xwOBA_30d_matchup_adj",
    off_col_away: str = "away_off_xwOBA_30d_matchup_adj",
) -> pd.DataFrame:
    """Rolling 30d mean lineup xwOBA norm per team (home/away split) for deltas."""
    g = history_games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    rows: list[pd.DataFrame] = []
    for side, team_col, off_col in (
        ("home", "home_id", off_col_home),
        ("away", "away_id", off_col_away),
    ):
        if off_col not in g.columns:
            continue
        sub = g[["game_id", "game_date", team_col, off_col]].dropna()
        sub = sub.rename(columns={team_col: "team_id", off_col: "off_xwoba"})
        sub["side"] = side
        rows.append(sub)
    if not rows:
        out = games.copy()
        out["home_off_norm_30d"] = np.nan
        out["away_off_norm_30d"] = np.nan
        return out

    long = pd.concat(rows, ignore_index=True).sort_values(["team_id", "side", "game_date"])
    norms: list[pd.DataFrame] = []
    for (team_id, side), grp in long.groupby(["team_id", "side"], sort=False):
        grp = grp.sort_values("game_date").set_index("game_date")
        roll = grp["off_xwoba"].rolling(f"{rolling_days}D", closed="left").mean()
        piece = grp.reset_index()
        piece["off_norm_30d"] = roll.to_numpy()
        piece["team_id"] = team_id
        piece["side"] = side
        norms.append(piece)

    norm_df = pd.concat(norms, ignore_index=True)
    out = games.copy()
    home_norm = norm_df[norm_df["side"] == "home"][
        ["game_id", "team_id", "off_norm_30d"]
    ].rename(columns={"off_norm_30d": "home_off_norm_30d"})
    away_norm = norm_df[norm_df["side"] == "away"][
        ["game_id", "team_id", "off_norm_30d"]
    ].rename(columns={"off_norm_30d": "away_off_norm_30d"})
    out = out.merge(home_norm, left_on=["game_id", "home_id"], right_on=["game_id", "team_id"], how="left")
    out = out.drop(columns=["team_id"], errors="ignore")
    out = out.merge(away_norm, left_on=["game_id", "away_id"], right_on=["game_id", "team_id"], how="left")
    out = out.drop(columns=["team_id"], errors="ignore")
    return out
