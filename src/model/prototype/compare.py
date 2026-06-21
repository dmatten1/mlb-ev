"""Offline comparison: production Pythag vs NB vs team-RS baseline prototypes.

Writes a markdown report to ``data/research/`` — never touches bet log or
production inference.

Usage::

    python -m src.model.prototype.compare --train-years 2023 2024 --test-year 2025
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.evaluate import calibration_table, pythag_metrics
from src.model.prototype.pipeline import (
    DEFAULT_PROTOTYPE_PATH,
    predict_prototype,
    save_prototype,
    train_prototype,
)
from src.model.prototype.tail_calibrator import tail_calibration_table
from src.model.prototype.tail_blend import (
    apply_run_level_blend,
    apply_tail_blend,
    fit_run_level_blend,
    fit_tail_blend,
    tail_team_mask,
)
from src.model.runs_model import DEFAULT_HFA_RUNS_BONUS
from src.model.prototype.team_baseline_model import (
    DEFAULT_TEAM_BASELINE_PATH,
    predict_team_baseline_runs,
    save_team_baseline_model,
    train_team_baseline_model,
)
from src.model.runs_model import BULLPEN_FEATURE_COLS, predict_runs, train_runs_model

logger = logging.getLogger("model.prototype.compare")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES_DIR = REPO_ROOT / "data/features"
DEFAULT_OUT = REPO_ROOT / "data/research/prototype_compare.md"
DEFAULT_BLEND_PATH = REPO_ROOT / "data/models/tail_blend_v1.pkl"

TAIL_TEAMS = ("Rockies", "Dodgers", "Colorado Rockies", "Los Angeles Dodgers")


def _load_year_parquet(year: int, features_dir: Path) -> pd.DataFrame:
    path = features_dir / f"training_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_parquet(path)


def _ensure_home_win(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_win" not in out.columns:
        out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)
    return out


def _metrics_block(name: str, p_home: np.ndarray, y: np.ndarray) -> list[str]:
    m = pythag_metrics(y, p_home)
    tail = tail_calibration_table(p_home, y)
    lines = [
        f"### {name}",
        f"- n={m.n}, accuracy={m.accuracy:.3%}, log_loss={m.log_loss:.4f}, "
        f"Brier={m.brier:.4f}",
        f"- mean p_home={m.mean_p_home:.3f} vs actual {m.home_win_rate:.3%}",
        "",
        "| region | n | p_mean | actual | gap |",
        "|--------|---|--------|--------|-----|",
    ]
    for _, row in tail.iterrows():
        lines.append(
            f"| {row['region']} | {int(row['n'])} | {row['p_mean']} | "
            f"{row['actual_rate']} | {row['gap']:+.4f} |"
        )
    lines.append("")
    return lines


def _team_involvement_mask(df: pd.DataFrame, needle: str) -> np.ndarray:
    for col in ("home_name", "away_name"):
        if col in df.columns:
            hit = df[col].astype(str).str.contains(needle, case=False, na=False)
            if hit.any():
                return hit.to_numpy()
    return np.zeros(len(df), dtype=bool)


def _subset_metrics_line(
    label: str,
    mask: np.ndarray,
    p: np.ndarray,
    y: np.ndarray,
) -> str:
    if not mask.any():
        return f"*{label}: no games*"
    m = pythag_metrics(y[mask], p[mask])
    actual = float(y[mask].mean())
    pred = float(p[mask].mean())
    return (
        f"**{label}** ({int(mask.sum())} games): "
        f"log_loss={m.log_loss:.4f}, pred={pred:.3f}, actual={actual:.3f}, "
        f"gap={actual - pred:+.3f}"
    )


def _team_bias_section(
    df: pd.DataFrame,
    p_baseline: np.ndarray,
    p_proto: np.ndarray,
    y: np.ndarray,
    *,
    p_team: np.ndarray | None = None,
    p_blend: np.ndarray | None = None,
) -> list[str]:
    lines = ["## Team tail bias (Rockies / Dodgers)", ""]
    for label, needle in (("Rockies", "Rockies"), ("Dodgers", "Dodgers")):
        mask = _team_involvement_mask(df, needle)
        if not mask.any():
            lines.append(f"*{label}: no games in test set*")
            continue
        actual = float(y[mask].mean())
        base = float(p_baseline[mask].mean())
        proto = float(p_proto[mask].mean())
        line = (
            f"**{label}** ({int(mask.sum())} games): "
            f"actual home-win={actual:.3f}, "
            f"baseline p={base:.3f} (gap {actual - base:+.3f}), "
            f"NB+cal p={proto:.3f} (gap {actual - proto:+.3f})"
        )
        if p_team is not None:
            team_p = float(p_team[mask].mean())
            line += f", team-RS p={team_p:.3f} (gap {actual - team_p:+.3f})"
        if p_blend is not None:
            blend_p = float(p_blend[mask].mean())
            line += f", blend p={blend_p:.3f} (gap {actual - blend_p:+.3f})"
        lines.append(line)
    lines.append("")
    return lines


def _eligible_metrics_section(
    eligible: np.ndarray,
    p_base: np.ndarray,
    p_team: np.ndarray,
    y: np.ndarray,
    *,
    min_games: int,
) -> list[str]:
    lines = [
        f"## Metrics — bet-eligible only (both teams ≥{min_games} season games played)",
        "",
    ]
    if not eligible.any():
        lines.append("*No eligible games in test set.*")
        lines.append("")
        return lines
    lines.extend(_metrics_block("Baseline (eligible)", p_base[eligible], y[eligible]))
    lines.extend(_metrics_block("Team RS baseline (eligible)", p_team[eligible], y[eligible]))
    lines.append(f"Eligible games: {int(eligible.sum())} / {len(eligible)}")
    lines.append("")
    return lines


def run_compare(
    *,
    train_years: list[int],
    test_year: int,
    features_dir: Path = DEFAULT_FEATURES_DIR,
    out_path: Path = DEFAULT_OUT,
    save_model_path: Path | None = DEFAULT_PROTOTYPE_PATH,
    n_sim: int = 8000,
) -> Path:
    """Train prototype, evaluate on test year, write markdown report."""
    train_frames = [_ensure_home_win(_load_year_parquet(y, features_dir)) for y in train_years]
    train = pd.concat(train_frames, ignore_index=True)
    test = _ensure_home_win(_load_year_parquet(test_year, features_dir))

    # Chronological cal slice from train (last 20%).
    gd = pd.to_datetime(train["game_date"])
    cutoff = gd.quantile(0.80)
    cal = train[gd >= cutoff].copy()
    fit_train = train[gd < cutoff].copy()

    logger.info("Training prototype on %d games (cal=%d)", len(fit_train), len(cal))
    proto = train_prototype(fit_train, cal, n_sim=n_sim)

    if save_model_path is not None:
        save_prototype(proto, save_model_path)
        logger.info("Saved prototype to %s", save_model_path)

    logger.info("Training team-RS baseline on %d games", len(fit_train))
    team_model = train_team_baseline_model(fit_train, history_games=fit_train)
    save_team_baseline_model(team_model, DEFAULT_TEAM_BASELINE_PATH)

    prod_model = train_runs_model(fit_train, BULLPEN_FEATURE_COLS)
    from src.model.evaluate import pythag_win_prob

    def _prod_p(games: pd.DataFrame) -> np.ndarray:
        pr = predict_runs(prod_model, games)
        return pythag_win_prob(
            pr["home_runs_pred"].to_numpy(),
            pr["away_runs_pred"].to_numpy(),
        )

    cal_team = predict_team_baseline_runs(team_model, cal, history_games=fit_train)
    cal_prod_runs = predict_runs(prod_model, cal)
    y_cal = cal["home_win"].astype(float).to_numpy()

    league_avg_raw = float(cal_team["league_avg_raw_runs"].iloc[0]) if "league_avg_raw_runs" in cal_team.columns else 4.5
    blend_model = fit_run_level_blend(
        cal_prod_runs["home_runs_pred"].to_numpy(),
        cal_prod_runs["away_runs_pred"].to_numpy(),
        cal_team["home_runs_pred_tb"].to_numpy(),
        cal_team["away_runs_pred_tb"].to_numpy(),
        cal_team["home_off_baseline"].to_numpy(),
        cal_team["away_off_baseline"].to_numpy(),
        y_cal,
        league_avg=team_model.league_avg_runs,
        league_avg_raw=league_avg_raw,
        home_baseline_raw=cal_team["home_team_rs_raw_roll"].to_numpy(),
        away_baseline_raw=cal_team["away_team_rs_raw_roll"].to_numpy(),
        hfa_runs=DEFAULT_HFA_RUNS_BONUS,
    )
    import pickle
    DEFAULT_BLEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_BLEND_PATH.open("wb") as f:
        pickle.dump(blend_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Run-level blend weights: mid=%.2f elite=%.2f cellar=%.2f both=%.2f",
        blend_model.weight_middle,
        blend_model.weight_elite,
        blend_model.weight_cellar,
        blend_model.weight_both_tail,
    )

    prod_pred = predict_runs(prod_model, test)
    prod_pred["p_home_prod"] = pythag_win_prob(
        prod_pred["home_runs_pred"].to_numpy(),
        prod_pred["away_runs_pred"].to_numpy(),
    )

    pred = predict_prototype(test, proto)
    team_pred = predict_team_baseline_runs(team_model, test, history_games=train)

    y = test["home_win"].astype(float).to_numpy()
    p_base = prod_pred["p_home_prod"].to_numpy()
    p_nb = pred["p_home_nb"].to_numpy()
    p_cal = pred["p_home_proto_cal"].to_numpy()
    p_team = team_pred["p_home_tb"].to_numpy()
    p_blend = apply_run_level_blend(
        blend_model,
        prod_pred["home_runs_pred"].to_numpy(),
        prod_pred["away_runs_pred"].to_numpy(),
        team_pred["home_runs_pred_tb"].to_numpy(),
        team_pred["away_runs_pred_tb"].to_numpy(),
        team_pred["home_off_baseline"].to_numpy(),
        team_pred["away_off_baseline"].to_numpy(),
        home_baseline_raw=team_pred["home_team_rs_raw_roll"].to_numpy(),
        away_baseline_raw=team_pred["away_team_rs_raw_roll"].to_numpy(),
        hfa_runs=DEFAULT_HFA_RUNS_BONUS,
    )
    eligible = team_pred["prototype_bet_eligible"].fillna(False).to_numpy()
    elite_m, cellar_m, both_m = tail_team_mask(
        team_pred["home_off_baseline"].to_numpy(),
        team_pred["away_off_baseline"].to_numpy(),
        team_model.league_avg_runs,
        elite_margin=blend_model.elite_margin,
        cellar_margin=blend_model.cellar_margin,
    )

    lines = [
        "# Prototype model comparison (offline)",
        "",
        f"Train years: {train_years} ({len(fit_train):,} fit + {len(cal):,} cal)",
        f"Test year: {test_year} ({len(test):,} games)",
        f"NB dispersion α={proto.dispersion.alpha:.2f} "
        f"(n={proto.dispersion.train_n}, residual_var={proto.dispersion.residual_var_mean:.3f})",
        f"Team RS league avg (park-adj): {team_model.league_avg_runs:.3f}",
        f"Shrink: w=min(1, season_games/30) × rolling RS_30d + (1−w) × league avg; park via index_runs",
        "",
        "> Paper trading unchanged — production still uses Ridge → Pythag.",
        "",
        "## Metrics (all games)",
        "",
    ]
    lines.extend(_metrics_block("Production Ridge + Pythag", p_base, y))
    lines.extend(_metrics_block("Team RS baseline + Ridge residuals", p_team, y))
    lines.extend(_metrics_block("Run-level tail blend (prod + team RS)", p_blend, y))
    lines.extend(_metrics_block("NB Monte Carlo (same μ as prod fit)", p_nb, y))
    lines.extend(_metrics_block("NB + tail isotonic", p_cal, y))

    lines.append("## Tail matchup subsets (blend vs components)")
    lines.append("")
    lines.append(
        f"Run-level blend weights (fit on cal): middle={blend_model.weight_middle:.2f}, "
        f"elite={blend_model.weight_elite:.2f}, cellar={blend_model.weight_cellar:.2f}, "
        f"both-tail={blend_model.weight_both_tail:.2f}"
    )
    lines.append("")
    for label, mask in (
        ("Elite team involved", elite_m),
        ("Cellar team involved", cellar_m),
        ("Elite vs cellar", both_m),
    ):
        lines.append(_subset_metrics_line(f"{label} — production", mask, p_base, y))
        lines.append(_subset_metrics_line(f"{label} — team RS", mask, p_team, y))
        lines.append(_subset_metrics_line(f"{label} — blend", mask, p_blend, y))
        lines.append("")

    lines.append("## Quantile calibration (baseline vs prototype+cal)")
    lines.append("")
    cal_base = calibration_table(p_base, y, bins=10).reset_index()
    cal_proto = calibration_table(p_cal, y, bins=10).reset_index()
    merged = cal_base.merge(
        cal_proto, on="bucket", suffixes=("_base", "_proto"),
    )
    lines.append("| bucket | n | p_base | actual | gap_base | p_proto | gap_proto |")
    lines.append("|--------|---|--------|--------|----------|---------|-----------|")
    for _, row in merged.iterrows():
        lines.append(
            f"| {int(row['bucket'])} | {int(row['n_base'])} | "
            f"{row['p_home_mean_base']:.3f} | {row['actual_home_win_base']:.3f} | "
            f"{row['calibration_gap_base']:+.3f} | "
            f"{row['p_home_mean_proto']:.3f} | {row['calibration_gap_proto']:+.3f} |"
        )
    lines.append("")

    lines.extend(_eligible_metrics_section(
        eligible, p_base, p_team, y, min_games=team_model.baseline_config.min_games_full_weight,
    ))

    lines.extend(_team_bias_section(test, p_base, p_cal, y, p_team=p_team, p_blend=p_blend))

    lines.extend([
        "## Team RS baseline notes",
        "",
        "- Anchor: shrunk 30d park-adjusted RS (home/away split), not full intercept replacement",
        "- Ridge learns SP/lineup/BP **residuals** on top of anchor",
        "- `prototype_bet_eligible`: both teams have ≥30 season games played",
        "",
        "## Next steps",
        "",
        "- Opponent RA baseline on runs-allowed side",
        "- Per-park dispersion for NB branch",
        "- Market shrinkage at extreme probabilities",
        "",
        f"Artifacts: `{save_model_path}`, `{DEFAULT_TEAM_BASELINE_PATH}`, `{DEFAULT_BLEND_PATH}`",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote report to %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Compare baseline vs prototype (offline).")
    ap.add_argument("--train-years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-save", action="store_true", help="Skip writing prototype pickle")
    ap.add_argument("--n-sim", type=int, default=8000)
    args = ap.parse_args()

    save_path = None if args.no_save else DEFAULT_PROTOTYPE_PATH
    try:
        run_compare(
            train_years=args.train_years,
            test_year=args.test_year,
            features_dir=args.features_dir,
            out_path=args.out,
            save_model_path=save_path,
            n_sim=args.n_sim,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Need training parquets under %s. Run the feature pipeline first.",
            args.features_dir,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
