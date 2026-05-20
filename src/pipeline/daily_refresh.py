"""Daily morning refresh: pull fresh data + rebuild features + (optionally) predict.

Orchestrates the existing ingestion modules in correct dependency order:

    1.  outcomes  — yesterday's final scores from MLB-StatsAPI
    2.  boxscores — lineups + pitchers for any games we haven't fetched yet
    3.  statcast  — incremental pitch-level pull (today − last_pulled_date)
    4.  oaa       — current-year Outs Above Average snapshot
    5.  features  — rebuild ``data/features/training_<year>.parquet``
    6.  predict   — (optional) run model + odds → ``data/predictions/<date>.parquet``

Each step is idempotent and safe to re-run. Skip individual steps with
``--skip-<step>`` (e.g. ``--skip-statcast`` if you've already pulled today).

Designed to run once per morning at ~6-8 AM ET, before that day's slate
of games starts publishing odds and lineups.

For **frequent** updates (settle scores, refresh tonight's slate + bet log,
rebuild dashboard without rebuilding Statcast/features), use
``python -m src.pipeline.live_refresh`` or ``make live-refresh`` — see Makefile
and ``docker compose`` service ``live``.

CLI:
    # Default: refresh everything for the current year + run predictions
    python -m src.pipeline.daily_refresh

    # Specific year
    python -m src.pipeline.daily_refresh --year 2026

    # Just refresh data, don't predict
    python -m src.pipeline.daily_refresh --no-predict

    # Skip slow steps if you ran them already today
    python -m src.pipeline.daily_refresh --skip-statcast --skip-features
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"
DEFAULT_RUNS_MODEL_CACHE = REPO_ROOT / "data" / "models" / "runs_model_bullpen_cached.pkl"

logger = logging.getLogger("pipeline.daily_refresh")


@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    note: str = ""


@dataclass
class RefreshResult:
    steps: list[StepResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, seconds: float, note: str = "") -> None:
        self.steps.append(StepResult(name, ok, seconds, note))


def _run_step(name: str, fn: Callable[[], str | None],
              result: RefreshResult, *, fail_fast: bool = False) -> bool:
    logger.info("[%s] start", name)
    t0 = time.monotonic()
    try:
        note = fn() or ""
        elapsed = time.monotonic() - t0
        logger.info("[%s] ok in %.1fs — %s", name, elapsed, note)
        result.add(name, True, elapsed, note)
        return True
    except Exception as exc:  # noqa: BLE001 — keep pipeline going by default
        elapsed = time.monotonic() - t0
        logger.error("[%s] FAILED after %.1fs: %s", name, elapsed, exc)
        logger.error("%s", traceback.format_exc())
        result.add(name, False, elapsed, f"FAILED: {exc}")
        if fail_fast:
            raise
        return False


# ---------------------------------------------------------------------------
# Individual step implementations
# ---------------------------------------------------------------------------

def step_outcomes(year: int, lookback_days: int = 3) -> str:
    """Pull MLB outcomes for the last N days. Idempotent — `fetch_outcomes`
    writes one JSON per game date, overwriting whatever was there before.
    Lookback handles late-night final games and rescheduled games.
    """
    from src.ingest.fetch_outcomes import run_backfill, DEFAULT_LOCAL_ROOT

    end = date.today()
    start = end - timedelta(days=lookback_days)
    summary = run_backfill(start, end, local_root=DEFAULT_LOCAL_ROOT)
    return (f"days_processed={summary['days_processed']} "
            f"days_written={summary['days_written']} "
            f"total_games={summary['total_games']}")


def step_boxscores(year: int) -> str:
    """Pull boxscores for any games whose outcomes parquet has them but
    whose boxscore JSON file doesn't exist yet. Idempotent.
    """
    from src.ingest.fetch_boxscores import backfill_year

    summary = backfill_year(year, sleep_s=0.05)
    return (f"requested={summary['requested']} fetched={summary['fetched']} "
            f"skipped={summary['skipped']} errors={summary['errors']}")


def step_statcast(year: int) -> str:
    """Incremental statcast pull for the current year."""
    from src.ingest.fetch_statcast_history import update_season_incremental

    out = update_season_incremental(year)
    size_mb = out.stat().st_size / 1e6 if out.exists() else 0.0
    return f"wrote {out.name} ({size_mb:.1f} MB)"


def step_oaa(year: int) -> str:
    """Overwrite the current year's OAA parquet with the latest Savant snapshot."""
    from src.ingest.fetch_oaa import save_oaa

    out = save_oaa(year)
    return f"wrote {out.name}"


def step_schedule(year: int, lookahead_days: int = 1) -> str:
    """Pull schedule + probable pitchers for today (and the next ``lookahead_days``).

    Idempotent — each date overwrites its own JSON snapshot. Lookahead
    lets us preview tomorrow's slate too in case we're running late.
    """
    from src.ingest.fetch_schedule import run_backfill

    today = date.today()
    end = today + timedelta(days=lookahead_days)
    summary = run_backfill(today, end)
    return (f"days_written={summary['days_written']} "
            f"total_games={summary['total_games']}")


def step_lineups(year: int) -> str:
    """Flatten boxscores into the tidy lineups parquets that build_features reads."""
    from src.features.lineup_loader import rollup_to_parquet

    wide, long, _names = rollup_to_parquet(year)
    return f"wrote {wide.name} + {long.name}"


def step_features(year: int) -> str:
    """Rebuild ``training_<year>.parquet`` from current raw + reference data."""
    from src.features.build_features import build_training_features

    df = build_training_features(year)
    return f"built {len(df)} rows; date range {df.game_date.min().date()} -> {df.game_date.max().date()}"


def step_predict(
    year: int,
    predictions_root: Path,
    *,
    use_cached_model: bool = False,
    model_cache_path: Path | None = None,
) -> str:
    """Train on prior seasons, predict tonight's slate, write parquet + markdown.

    When ``use_cached_model`` is True and ``model_cache_path`` exists, skips
    training and loads the pickle (for frequent :mod:`live_refresh` runs). A
    full :func:`run_refresh` retrains by default and overwrites the cache.

    Priority order:
      1. Try the **projected** slate for today (uses
         ``build_projected_features`` against the schedule + projected
         lineups). This is what you want for a morning betting decision.
      2. If today's schedule isn't pulled or projection failed, fall
         back to the most recent date with both historical features and
         odds available (useful for backtest-style review).
      3. If neither works, return a "skipped" string and let the data
         refresh be considered the day's win.

    Returns a status string. Skipping is not treated as failure.
    """
    import pandas as pd
    from src.features.build_features import build_projected_features
    from src.model.runs_model import (
        BULLPEN_FEATURE_COLS,
        load_runs_model,
        save_runs_model,
        train_runs_model,
    )
    from src.inference.predict_slate import predict_slate
    from src.inference.odds_loader import (
        load_snapshots_long, best_lines_per_game,
        build_team_name_to_id, attach_team_ids,
    )

    today = date.today()
    predictions_root.mkdir(parents=True, exist_ok=True)

    cache_path = model_cache_path or DEFAULT_RUNS_MODEL_CACHE
    if use_cached_model and cache_path.exists():
        rm = load_runs_model(cache_path)
        logger.info(
            "[predict] loaded cached RunsModel from %s (train_n=%d)",
            cache_path, rm.train_n,
        )
    else:
        # Train on 2023 + 2024 (locked regime — see calibration analysis).
        train = pd.concat([
            pd.read_parquet(REPO_ROOT / "data/features/training_2023.parquet"),
            pd.read_parquet(REPO_ROOT / "data/features/training_2024.parquet"),
        ], ignore_index=True)
        rm = train_runs_model(train, BULLPEN_FEATURE_COLS)
        logger.info("[predict] trained Ridge on %d rows", len(train))
        try:
            save_runs_model(rm, cache_path)
            logger.info("[predict] wrote model cache to %s", cache_path)
        except OSError as e:
            logger.warning("[predict] could not save model cache: %s", e)

    team_name_map = build_team_name_to_id(
        REPO_ROOT / "data/features/training_2025.parquet"
    )

    def _odds_for(target: date):
        lo = (target - timedelta(days=1)).isoformat()
        hi = target.isoformat()
        odds_long = load_snapshots_long(date_lo=lo, date_hi=hi)
        if odds_long.empty:
            return None
        per_game = best_lines_per_game(odds_long, close_window_minutes=30,
                                       price_strategy="best")
        return attach_team_ids(per_game, team_name_map)

    def _write(slate, target_str: str, slate_label: str) -> str:
        pq_path = predictions_root / f"{target_str}.parquet"
        md_path = predictions_root / f"{target_str}.md"
        slate.to_parquet(pq_path, index=False)
        _write_markdown_summary(slate, md_path, run_date=target_str,
                                 slate_label=slate_label)
        n_bets = (slate.recommended.isin(["home", "away"])).sum()
        # Append/update the bet tracker. Only meaningful when this slate
        # is for "today" (commence_time is in the future / very recent);
        # for fallback historical slates, the bets are already settled
        # so logging would be misleading — we skip.
        bet_log_msg = ""
        if slate_label.startswith("tonight"):
            try:
                from src.tracking.bet_log import log_recommendations
                res = log_recommendations(slate)
                detail_parts: list[str] = []
                if res["inserted"] or res["updated"]:
                    detail_parts.append(
                        f"+{res['inserted']} new, {res['updated']} updated"
                    )
                spl = res.get("skipped_paper_locked", 0)
                if spl:
                    detail_parts.append(f"{spl} skipped (paper lock)")
                if detail_parts:
                    bet_log_msg = "; bet_log: " + ", ".join(detail_parts)
            except Exception as e:  # noqa: BLE001
                logger.warning("[predict] bet_log update failed: %s", e)
        return (f"wrote {pq_path.name} + {md_path.name} ({slate_label}): "
                f"{len(slate)} games, {n_bets} bets recommended{bet_log_msg}")

    # 1. Try today's projected slate first.
    try:
        proj = build_projected_features(today, year=year)
    except FileNotFoundError as e:
        logger.info("[predict] projected slate unavailable: %s", e)
        proj = pd.DataFrame()
    except Exception as e:  # noqa: BLE001
        logger.warning("[predict] projected slate build failed: %s — falling back.", e)
        proj = pd.DataFrame()

    if not proj.empty:
        odds = _odds_for(today)
        if odds is not None and not odds.empty:
            slate = predict_slate(proj, rm, odds)
            if not slate.empty:
                return _write(slate, today.isoformat(),
                              slate_label="tonight (projected lineups)")

    # 2. Fall back to most recent historical slate.
    feat_path = REPO_ROOT / f"data/features/training_{year}.parquet"
    games = pd.read_parquet(feat_path)
    feature_dates = sorted({d.date() for d in pd.to_datetime(games.game_date)})
    for target in reversed([d for d in feature_dates if d <= today]):
        slate_games = games[games.game_date.dt.date == target]
        if slate_games.empty:
            continue
        odds = _odds_for(target)
        if odds is None or odds.empty:
            continue
        slate = predict_slate(slate_games, rm, odds)
        if slate.empty:
            continue
        return _write(slate, target.isoformat(),
                      slate_label=f"historical fallback ({target.isoformat()})")

    return ("skipped — no projected slate ready and no historical date "
            "had matching odds. The morning data refresh still succeeded; "
            "predictions will resume once the schedule + odds are pulled.")


# ---------------------------------------------------------------------------
# Markdown summary writer (for at-a-glance morning review)
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%" if x == x else "  -  "


def _fmt_american(x: float) -> str:
    if x != x:
        return "  -  "
    return f"{x:+.0f}"


def _write_markdown_summary(slate, md_path: Path, *, run_date: str,
                             slate_label: str = "") -> None:
    import pandas as pd  # local import to keep module-level imports light

    n_games = len(slate)
    rec = slate[slate.recommended.isin(["home", "away"])].copy()
    n_bets = len(rec)
    total_stake_pct = rec.recommended_kelly.sum() * 100

    lines: list[str] = []
    title_suffix = f" ({slate_label})" if slate_label else ""
    lines.append(f"# MLB EV Predictions — {run_date}{title_suffix}\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"- Games on slate: **{n_games}**")
    lines.append(f"- Recommended bets: **{n_bets}**")
    lines.append(f"- Total Kelly (bankroll): **{total_stake_pct:.2f}%** — optional % of bankroll per play.\n")
    lines.append(
        "> **Tracker / units:** **risk_units** use the same pre–daily-cap Kelly as the model. "
        "Each slate defines **one unit** as the **mean** recommended Kelly that day (stored in the log as "
        "**risk_ref_kelly**), so a typical play is ~**1u at risk**; stronger/weaker edges spread toward **2u** / **0.5u** "
        "(hard clamp). Wins pay **risk_units × (to-win)**; losses **−risk_units**. "
        "**Kelly (bankroll %)** in the table is *after* the slate’s daily cap — that’s for real-dollar sizing.\n"
    )

    if n_bets:
        has_book = ("home_book" in rec.columns and "away_book" in rec.columns)
        lines.append("## Recommended bets\n")
        if has_book:
            lines.append("| Game | Side | Book | Odds | Model p | Fair p | Edge | EV | Kelly (bankroll %) |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| Game | Side | Odds | Model p | Fair p | Edge | EV | Kelly (bankroll %) |")
            lines.append("|---|---|---|---|---|---|---|---|")
        rec_sorted = rec.sort_values("recommended_ev", ascending=False)
        for _, r in rec_sorted.iterrows():
            matchup = f"{r.away_name} @ {r.home_name}"
            side_team = r.home_name if r.recommended == "home" else r.away_name
            side_label = "HOME" if r.recommended == "home" else "AWAY"
            odds = (r.home_price_american if r.recommended == "home"
                    else r.away_price_american)
            model_p = (r.p_home if r.recommended == "home" else 1 - r.p_home)
            fair_p = (r.home_fair_p if r.recommended == "home" else r.away_fair_p)
            edge = (r.edge_home if r.recommended == "home" else r.edge_away)
            ev = r.recommended_ev
            if has_book:
                book = (r.home_book if r.recommended == "home" else r.away_book) or "—"
                lines.append(
                    f"| {matchup} | {side_label} {side_team} | {book} | "
                    f"{_fmt_american(odds)} | {_fmt_pct(model_p)} | "
                    f"{_fmt_pct(fair_p)} | {edge*100:+.1f}pp | "
                    f"{ev*100:+.2f}% | {r.recommended_kelly*100:.2f}% |"
                )
            else:
                lines.append(
                    f"| {matchup} | {side_label} {side_team} | "
                    f"{_fmt_american(odds)} | {_fmt_pct(model_p)} | "
                    f"{_fmt_pct(fair_p)} | {edge*100:+.1f}pp | "
                    f"{ev*100:+.2f}% | {r.recommended_kelly*100:.2f}% |"
                )
        lines.append("")

    has_source = ("home_lineup_source" in slate.columns
                  and "away_lineup_source" in slate.columns)
    lines.append("## Full slate (model probabilities)\n")
    header_cols = ["Game", "Lineups", "Away xR", "Home xR",
                    "Model p_home", "Fair p_home", "Verdict"]
    if not has_source:
        header_cols = [c for c in header_cols if c != "Lineups"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "---|" * len(header_cols))
    for _, r in slate.sort_values("commence_time").iterrows():
        matchup = f"{r.away_name} @ {r.home_name}"
        verdict = ("bet " + r.recommended.upper()) if r.recommended in ("home","away") else "no bet"
        if has_source:
            # 'A' = actual lineup posted; 'P' = projected; mixed unusual.
            tag_a = "A" if r.away_lineup_source == "actual" else "P"
            tag_h = "A" if r.home_lineup_source == "actual" else "P"
            src_cell = f"{tag_a}/{tag_h}"
            lines.append(
                f"| {matchup} | {src_cell} | {r.away_runs_pred:.2f} | "
                f"{r.home_runs_pred:.2f} | {_fmt_pct(r.p_home)} | "
                f"{_fmt_pct(r.home_fair_p)} | {verdict} |"
            )
        else:
            lines.append(
                f"| {matchup} | {r.away_runs_pred:.2f} | {r.home_runs_pred:.2f} | "
                f"{_fmt_pct(r.p_home)} | {_fmt_pct(r.home_fair_p)} | {verdict} |"
            )
    if has_source:
        lines.append("")
        lines.append("Lineups column: `A` = actual lineup posted by team, "
                      "`P` = projected (modal-mode estimator, IL-filtered).")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tracking: CLV reconciliation + outcomes -> HTML dashboard
# ---------------------------------------------------------------------------

def step_track() -> str:
    """Reconcile CLV + outcomes for the bet log, then rebuild dashboard.

    Cheap to run every refresh — the log is small, CLV/outcome queries
    only touch unfinished rows, and the HTML render is ~ms.
    """
    from src.tracking.bet_log import reconcile_clv, reconcile_outcomes
    from src.tracking.dashboard import render

    clv = reconcile_clv()
    outc = reconcile_outcomes(
        outcomes_root=REPO_ROOT / "data/outcomes",
        features_root=REPO_ROOT / "data/features",
    )
    out_path = render()
    return (f"bet log: clv +{clv['computed']} ({clv['skipped']} pending), "
            f"outcomes +{outc['resolved']} ({outc['pending']} pending) — "
            f"wrote {out_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_refresh(
    year: int,
    *,
    do_outcomes: bool = True,
    do_boxscores: bool = True,
    do_statcast: bool = True,
    do_oaa: bool = True,
    do_lineups: bool = True,
    do_features: bool = True,
    do_schedule: bool = True,
    do_predict: bool = True,
    do_track: bool = True,
    predictions_root: Path = DEFAULT_PREDICTIONS_ROOT,
    predict_use_cached_model: bool = False,
    model_cache_path: Path | None = None,
    fail_fast: bool = False,
) -> RefreshResult:
    """Run the daily pipeline end-to-end. Each step is wrapped so a
    transient failure in one stage doesn't kill the whole job (use
    ``fail_fast=True`` to flip that).
    """
    result = RefreshResult()
    t0 = time.monotonic()
    logger.info("Daily refresh starting | year=%d | %s",
                year, datetime.now().isoformat(timespec="seconds"))

    if do_outcomes:
        _run_step("outcomes", lambda: step_outcomes(year), result, fail_fast=fail_fast)
    if do_boxscores:
        _run_step("boxscores", lambda: step_boxscores(year), result, fail_fast=fail_fast)
    if do_statcast:
        _run_step("statcast", lambda: step_statcast(year), result, fail_fast=fail_fast)
    if do_oaa:
        _run_step("oaa", lambda: step_oaa(year), result, fail_fast=fail_fast)
    if do_lineups:
        _run_step("lineups", lambda: step_lineups(year), result, fail_fast=fail_fast)
    if do_features:
        _run_step("features", lambda: step_features(year), result, fail_fast=fail_fast)
    if do_schedule:
        _run_step("schedule", lambda: step_schedule(year), result, fail_fast=fail_fast)
    if do_predict:
        mcp = model_cache_path
        pcm = predict_use_cached_model
        _run_step(
            "predict",
            lambda: step_predict(
                year, predictions_root,
                use_cached_model=pcm,
                model_cache_path=mcp,
            ),
            result,
            fail_fast=fail_fast,
        )
    if do_track:
        _run_step("track", lambda: step_track(),
                  result, fail_fast=fail_fast)

    total = time.monotonic() - t0
    n_ok = sum(1 for s in result.steps if s.ok)
    n_total = len(result.steps)
    logger.info("Daily refresh done in %.1fs | %d/%d steps ok", total, n_ok, n_total)
    for s in result.steps:
        flag = "OK " if s.ok else "FAIL"
        logger.info("  [%s] %-10s %6.1fs  %s", flag, s.name, s.seconds, s.note)
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the daily MLB EV refresh pipeline.")
    p.add_argument("--year", type=int, default=date.today().year)
    p.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    p.add_argument("--fail-fast", action="store_true",
                   help="Abort on first step failure (default: log and continue).")
    p.add_argument("--skip-outcomes",  action="store_true")
    p.add_argument("--skip-boxscores", action="store_true")
    p.add_argument("--skip-statcast",  action="store_true")
    p.add_argument("--skip-oaa",       action="store_true")
    p.add_argument("--skip-lineups",   action="store_true")
    p.add_argument("--skip-features",  action="store_true")
    p.add_argument("--skip-schedule",  action="store_true")
    p.add_argument("--skip-predict",   action="store_true")
    p.add_argument("--skip-track",     action="store_true")
    p.add_argument("--no-predict", action="store_true",
                   help="Alias for --skip-predict (refresh data only).")
    p.add_argument("--predict-use-cached-model", action="store_true",
                   help="Load cached Ridge model if present (frequent/sidecar runs).")
    p.add_argument("--model-cache", type=Path, default=None,
                   help=f"Override pickle path (default: {DEFAULT_RUNS_MODEL_CACHE})")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    result = run_refresh(
        year=args.year,
        do_outcomes=not args.skip_outcomes,
        do_boxscores=not args.skip_boxscores,
        do_statcast=not args.skip_statcast,
        do_oaa=not args.skip_oaa,
        do_lineups=not args.skip_lineups,
        do_features=not args.skip_features,
        do_schedule=not args.skip_schedule,
        do_track=not args.skip_track,
        do_predict=not (args.skip_predict or args.no_predict),
        predictions_root=args.predictions_root,
        predict_use_cached_model=args.predict_use_cached_model,
        model_cache_path=args.model_cache,
        fail_fast=args.fail_fast,
    )
    n_failed = sum(1 for s in result.steps if not s.ok)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
