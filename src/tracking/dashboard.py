"""Static HTML dashboard rendered from the bet log.

Renders a single self-contained ``bet_dashboard.html`` with two tabs:

**Bets**
1. **Summary cards** — overall stats (bets settled, wins, profit in
   units, ROI per bet, hit rate, average CLV, CLV beat rate).
2. **Bankroll trajectory** — cumulative profit-per-unit over time, with
   per-day markers (Chart.js, fed inline as a JSON array).
3. **Bet table** — every recommended bet with date,
   matchup (pending rows embed probable SP, e.g. ``Brewers (Brown) @ Cubs (Smith)``),
   **live score** for pending games (MLB Stats API),
   book, recommended side, model p, fair p at rec,
   CLV (pp), outcome, P/L.
   Sortable + filterable in-browser via a tiny vanilla JS script.

**Stats**
- Home / away splits (record, P/L, ROI)
- P/L histograms by odds bucket, bet size, predicted EV, and model win p (settled bets)
- Per-team record, P/L, hit rate, and calibration gap (sortable table)

Re-rendered on every ``live_refresh`` / ``step_track`` run (each odds pull).

No external CSS/JS files: Chart.js is loaded from a CDN inside the
HTML. Open the file in any browser.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import html
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.tracking.bet_log import (
    DEFAULT_LOG_PATH,
    filter_log_by_season,
    load_log,
    summarize_frame,
)
from src.tracking.live_scores import format_live_score, live_scores_for_pending_log

logger = logging.getLogger("tracking.dashboard")


DEFAULT_OUT = Path("data/tracking/bet_dashboard.html")
PAPER_LINEUP_POLICY_NOTE = "actual lineups only as of 5/23"

# Hold-out / live test year: summary + chart + table use Kelly-scaled rows for
# this UTC commence year only. Set to ``None`` to include every season in the log.
DEFAULT_DASHBOARD_SEASON_YEAR: int | None = 2026

# Ordered American-odds buckets for the stats histogram (settled bets).
# (internal key, display label)
ODDS_BUCKET_ORDER: list[tuple[str, str]] = [
    ("heavy_fav", "Heavy Favorite (−151+)"),
    ("fav", "Favorite (−111 to −150)"),
    ("pickem", "Pick'em"),
    ("underdog", "Underdog (+100 to +149)"),
    ("big_dog", "Big Dog (+150+)"),
]

# Kelly-scaled units at risk (settled bets).
STAKE_BUCKET_ORDER: list[tuple[str, str]] = [
    ("small", "0.5–0.99u"),
    ("medium", "1–1.49u"),
    ("large", "1.5–2u"),
]

# Predicted EV at recommendation (settled bets). 1% bins — finer than stake buckets.
EV_BUCKET_WIDTH = 0.01

# Model win probability at recommendation (settled bets).
MODEL_P_BUCKET_ORDER: list[tuple[str, str]] = [
    ("under_35", "<35%"),
    ("p35_45", "35–45%"),
    ("p45_55", "45–55%"),
    ("p55_65", "55–65%"),
    ("p65_75", "65–75%"),
    ("over_75", "≥75%"),
]


def _odds_bucket_key(odds: float | None) -> str | None:
    if odds is None or pd.isna(odds):
        return None
    o = float(odds)
    if o >= 150:
        return "big_dog"
    if o >= 100:
        return "underdog"
    if o >= -110:
        return "pickem"
    if o >= -150:
        return "fav"
    return "heavy_fav"


def _stake_bucket_key(risk_units: float | None) -> str | None:
    if risk_units is None or pd.isna(risk_units):
        return None
    u = float(risk_units)
    if u < 0.5 or u > 2.0:
        return None
    if u < 1.0:
        return "small"
    if u < 1.5:
        return "medium"
    return "large"


def _ev_bucket_index(ev: float | None) -> int | None:
    if ev is None or pd.isna(ev):
        return None
    e = float(ev)
    if e < 0:
        return None
    return int(e / EV_BUCKET_WIDTH)


def _ev_bucket_label(idx: int) -> str:
    lo = idx * EV_BUCKET_WIDTH
    hi = lo + EV_BUCKET_WIDTH
    return f"{lo * 100:.0f}–{hi * 100:.0f}%"


def _model_p_bucket_key(model_p: float | None) -> str | None:
    if model_p is None or pd.isna(model_p):
        return None
    p = float(model_p)
    if p < 0.35:
        return "under_35"
    if p < 0.45:
        return "p35_45"
    if p < 0.55:
        return "p45_55"
    if p < 0.65:
        return "p55_65"
    if p < 0.75:
        return "p65_75"
    return "over_75"


def _prepare_log_frame(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return log
    out = log.copy()
    out["commence_time"] = pd.to_datetime(out["commence_time"], utc=True)
    out["odds_at_rec"] = pd.to_numeric(out["odds_at_rec"], errors="coerce")
    if "model_p" in out.columns:
        out["model_p"] = pd.to_numeric(out["model_p"], errors="coerce")
    out["profit_units"] = pd.to_numeric(out["profit_units"], errors="coerce")
    out["risk_units"] = pd.to_numeric(out["risk_units"], errors="coerce").fillna(1.0)
    if "ev_at_rec" in out.columns:
        out["ev_at_rec"] = pd.to_numeric(out["ev_at_rec"], errors="coerce")
    out["odds_bucket"] = out["odds_at_rec"].map(_odds_bucket_key)
    out["stake_bucket"] = out["risk_units"].map(_stake_bucket_key)
    if "ev_at_rec" in out.columns:
        out["ev_bucket"] = out["ev_at_rec"].map(_ev_bucket_index)
    if "model_p" in out.columns:
        out["model_p_bucket"] = out["model_p"].map(_model_p_bucket_key)
    return out


def _aggregate_team_stats(frame: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    """Per-team settled record/P/L grouped by *group_col* (pending excluded)."""
    settled = frame[frame["outcome"].isin(["won", "lost", "push"])]
    rows: list[dict[str, Any]] = []
    for team, s in settled.groupby(group_col, sort=False):
        wins = int((s["outcome"] == "won").sum())
        losses = int((s["outcome"] == "lost").sum())
        pl = float(s["profit_units"].fillna(0).sum())
        risk = float(s["risk_units"].sum())
        avg_win_p = float(s["model_p"].mean()) if len(s) and s["model_p"].notna().any() else None
        decided = wins + losses
        hit_rate = (wins / decided) if decided > 0 else None
        calib_gap_pp = (
            (avg_win_p - hit_rate) * 100.0
            if avg_win_p is not None and hit_rate is not None
            else None
        )
        rows.append({
            "team": str(team),
            "bets": int(len(s)),
            "wins": wins,
            "losses": losses,
            "profit_units": pl,
            "risk_units": risk,
            "roi_pct": (100.0 * pl / risk) if risk > 0 else 0.0,
            "avg_win_p": avg_win_p,
            "hit_rate": hit_rate,
            "calib_gap_pp": calib_gap_pp,
        })
    rows.sort(key=lambda r: (r["profit_units"], r["bets"]), reverse=True)
    return rows


def compute_bet_stats(log: pd.DataFrame) -> dict[str, Any]:
    """Aggregate team, side, and odds-bucket stats for the Stats tab."""
    empty_side = {
        "bets": 0, "wins": 0, "losses": 0, "profit_units": 0.0,
        "risk_units": 0.0, "roi_pct": 0.0, "avg_win_p": None,
    }
    if log.empty:
        return {
            "teams_for": [],
            "teams_against": [],
            "sides": {"home": dict(empty_side), "away": dict(empty_side)},
            "odds_buckets": [],
            "stake_buckets": [],
            "ev_buckets": [],
            "model_p_buckets": [],
        }

    frame = _prepare_log_frame(log)
    frame["opponent"] = frame.apply(
        lambda r: r["away_name"] if r["recommended_side"] == "home" else r["home_name"],
        axis=1,
    )
    settled = frame[frame["outcome"].isin(["won", "lost", "push"])].copy()

    teams_for = _aggregate_team_stats(frame, "recommended_team")
    teams_against = _aggregate_team_stats(frame, "opponent")

    sides: dict[str, dict[str, Any]] = {}
    for side in ("home", "away"):
        s = settled[settled["recommended_side"] == side]
        wins = int((s["outcome"] == "won").sum())
        losses = int((s["outcome"] == "lost").sum())
        pl = float(s["profit_units"].fillna(0).sum())
        risk = float(s["risk_units"].sum())
        avg_win_p = float(s["model_p"].mean()) if len(s) and s["model_p"].notna().any() else None
        sides[side] = {
            "bets": int(len(s)),
            "wins": wins,
            "losses": losses,
            "profit_units": pl,
            "risk_units": risk,
            "roi_pct": (100.0 * pl / risk) if risk > 0 else 0.0,
            "avg_win_p": avg_win_p,
        }

    odds_buckets: list[dict[str, Any]] = []
    for key, label in ODDS_BUCKET_ORDER:
        s = settled[settled["odds_bucket"] == key]
        wins = int((s["outcome"] == "won").sum())
        losses = int((s["outcome"] == "lost").sum())
        pl = float(s["profit_units"].fillna(0).sum())
        risk = float(s["risk_units"].sum())
        odds_buckets.append({
            "label": label,
            "key": key,
            "bets": int(len(s)),
            "wins": wins,
            "losses": losses,
            "profit_units": pl,
            "risk_units": risk,
            "roi_pct": (100.0 * pl / risk) if risk > 0 else 0.0,
        })

    stake_buckets: list[dict[str, Any]] = []
    for key, label in STAKE_BUCKET_ORDER:
        s = settled[settled["stake_bucket"] == key]
        wins = int((s["outcome"] == "won").sum())
        losses = int((s["outcome"] == "lost").sum())
        pl = float(s["profit_units"].fillna(0).sum())
        risk = float(s["risk_units"].sum())
        stake_buckets.append({
            "label": label,
            "key": key,
            "bets": int(len(s)),
            "wins": wins,
            "losses": losses,
            "profit_units": pl,
            "risk_units": risk,
            "roi_pct": (100.0 * pl / risk) if risk > 0 else 0.0,
        })

    ev_buckets: list[dict[str, Any]] = []
    if "ev_bucket" in settled.columns:
        for idx in sorted(settled["ev_bucket"].dropna().astype(int).unique()):
            s = settled[settled["ev_bucket"] == idx]
            wins = int((s["outcome"] == "won").sum())
            losses = int((s["outcome"] == "lost").sum())
            pl = float(s["profit_units"].fillna(0).sum())
            risk = float(s["risk_units"].sum())
            ev_buckets.append({
                "label": _ev_bucket_label(int(idx)),
                "key": int(idx),
                "bets": int(len(s)),
                "wins": wins,
                "losses": losses,
                "profit_units": pl,
                "risk_units": risk,
                "roi_pct": (100.0 * pl / risk) if risk > 0 else 0.0,
            })

    model_p_buckets: list[dict[str, Any]] = []
    if "model_p_bucket" in settled.columns:
        for key, label in MODEL_P_BUCKET_ORDER:
            s = settled[settled["model_p_bucket"] == key]
            wins = int((s["outcome"] == "won").sum())
            losses = int((s["outcome"] == "lost").sum())
            pl = float(s["profit_units"].fillna(0).sum())
            risk = float(s["risk_units"].sum())
            model_p_buckets.append({
                "label": label,
                "key": key,
                "bets": int(len(s)),
                "wins": wins,
                "losses": losses,
                "profit_units": pl,
                "risk_units": risk,
                "roi_pct": (100.0 * pl / risk) if risk > 0 else 0.0,
            })

    return {
        "teams_for": teams_for,
        "teams_against": teams_against,
        "sides": sides,
        "odds_buckets": odds_buckets,
        "stake_buckets": stake_buckets,
        "ev_buckets": ev_buckets,
        "model_p_buckets": model_p_buckets,
    }


def _fmt_money(units: float) -> str:
    sign = "+" if units >= 0 else "−"
    return f"{sign}{abs(units):.2f}u"


def _fmt_pct(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "—"
    return f"{float(p) * 100:.1f}%"


def _fmt_pp_gap(gap_pp: float | None) -> str:
    if gap_pp is None or pd.isna(gap_pp):
        return "—"
    g = float(gap_pp)
    sign = "+" if g >= 0 else "−"
    return f"{sign}{abs(g):.1f}pp"


def _fmt_pp(pp: float | None) -> str:
    if pp is None or pd.isna(pp):
        return "—"
    sign = "+" if pp >= 0 else "−"
    return f"{sign}{abs(pp):.2f}pp"


def _fmt_american(odds: float | None) -> str:
    if odds is None or pd.isna(odds):
        return "—"
    o = int(round(float(odds)))
    return f"+{o}" if o > 0 else f"{o}"


def _outcome_class(outcome: str | None) -> str:
    return {"won": "won", "lost": "lost", "push": "push",
            "pending": "pending", "void": "void"}.get(str(outcome), "")


def _matchup_team_display(name: str) -> str:
    """Title-case team nickname for matchup cells."""
    t = name.strip()
    if not t:
        return t
    return t.title()


def _pitcher_surname_display(full_name: str) -> str:
    """Last name token, capitalized; ``TBD`` if unknown."""
    s = full_name.strip()
    if not s:
        return "TBD"
    parts = s.split()
    token = parts[-1] if parts else s
    return token.capitalize()


def _pitcher_names_from_game(
    away_name: Any,
    home_name: Any,
) -> tuple[str, str]:
    aa = str(away_name).strip() if pd.notna(away_name) and away_name is not None else ""
    hh = str(home_name).strip() if pd.notna(home_name) and home_name is not None else ""
    return aa, hh


def _pitcher_lookup_from_schedule_snapshots(
    log: pd.DataFrame,
) -> dict[int, tuple[str, str]]:
    """``game_id`` → (away probable SP, home probable SP) for pending bets.

    Reads cached schedule JSON when present; falls back to a live StatsAPI
    pull for any pending ``game_id`` still missing probable-pitcher names
    (common on Lambda when today's schedule snapshot is not on disk yet).
    """
    lookup: dict[int, tuple[str, str]] = {}
    if log.empty or "game_date" not in log.columns:
        return lookup

    pend = log[log["outcome"].astype(str) == "pending"]
    if pend.empty:
        return lookup

    from src.ingest.fetch_schedule import fetch_schedule_for_date, load_schedule_for_date
    import src.ingest.fetch_schedule as fs

    pending_by_date: dict[date, set[int]] = {}
    for _, row in pend.iterrows():
        try:
            d = pd.Timestamp(row["game_date"]).normalize().date()
            pending_by_date.setdefault(d, set()).add(int(row["game_id"]))
        except Exception:
            continue

    def _store(gid: int, away: Any, home: Any) -> None:
        lookup[gid] = _pitcher_names_from_game(away, home)

    for d in sorted(pending_by_date):
        pending_ids = pending_by_date[d]
        sdf = load_schedule_for_date(d, local_root=fs.DEFAULT_LOCAL_ROOT)
        if not sdf.empty and (
            "away_probable_pitcher_name" in sdf.columns
            and "home_probable_pitcher_name" in sdf.columns
        ):
            for _, row in sdf.iterrows():
                try:
                    gid = int(row["game_id"])
                except (TypeError, ValueError, KeyError):
                    continue
                if gid in pending_ids:
                    _store(
                        gid,
                        row.get("away_probable_pitcher_name"),
                        row.get("home_probable_pitcher_name"),
                    )

        missing = [
            gid for gid in pending_ids
            if not any(lookup.get(gid, ("", "")))
        ]
        if not missing:
            continue

        try:
            games = fetch_schedule_for_date(d)
        except Exception as exc:  # noqa: BLE001
            logger.warning("probable-pitcher live fetch failed for %s: %s", d, exc)
            continue

        for g in games:
            try:
                gid = int(g["game_id"])
            except (TypeError, ValueError, KeyError):
                continue
            if gid in pending_ids:
                _store(
                    gid,
                    g.get("away_probable_pitcher_name"),
                    g.get("home_probable_pitcher_name"),
                )

    return lookup


def render(
    log_path: Path | str = DEFAULT_LOG_PATH,
    out_path: Path | str = DEFAULT_OUT,
    *,
    season_year: int | None = DEFAULT_DASHBOARD_SEASON_YEAR,
) -> Path:
    """Render the dashboard and return the path.

    *season_year* restricts summary, cumulative P/L, and the bet table to games
    whose UTC ``commence_time`` falls in that calendar year (Kelly-scaled
    ``profit_units`` / ``risk_units`` are unchanged — only which rows appear).
    """
    full = load_log(log_path)
    full = filter_log_by_season(full, season_year)
    log = full[~full["outcome"].astype(str).eq("void")].reset_index(drop=True)
    summary = summarize_frame(log)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the trajectory: cumulative profit-per-unit ordered by commence_time.
    # Chart: category x-axis (Chart.js 'time' scale needs a date adapter; we
    # avoid extra CDN scripts by using labels).
    if not log.empty:
        log = log.copy()
        log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
        log = log.sort_values("commence_time")
        settled = log[log["outcome"].isin(["won", "lost", "push"])].copy()
        settled["profit_units"] = settled["profit_units"].astype(float).fillna(0.0)
        settled["cum_profit"] = settled["profit_units"].cumsum()
        labels: list[str] = []
        for i, ct in enumerate(settled["commence_time"]):
            ts = pd.Timestamp(ct).tz_convert("America/New_York")
            labels.append(ts.strftime("%b %d · ") + str(ts.hour).zfill(2) + ":" + str(ts.minute).zfill(2))
        values = [float(x) for x in settled["cum_profit"]]
    else:
        labels, values = [], []

    live_lookup, live_fetched_at = live_scores_for_pending_log(log)
    stats = compute_bet_stats(log)
    generated_at = datetime.now(timezone.utc)
    html_doc = _build_html(
        log, summary, labels, values,
        season_year=season_year,
        live_lookup=live_lookup,
        live_fetched_at=live_fetched_at,
        stats=stats,
        generated_at=generated_at,
    )
    out_path.write_text(html_doc, encoding="utf-8")
    logger.info("Wrote dashboard to %s", out_path)
    return out_path


def _build_html(
    log: pd.DataFrame,
    summary: dict,
    traj_labels: list[str],
    traj_values: list[float],
    *,
    season_year: int | None,
    live_lookup: dict | None = None,
    live_fetched_at: datetime | None = None,
    stats: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> str:
    panel_year = str(int(season_year)) if season_year is not None else "All seasons"
    cards = _render_summary_cards(summary)
    table = _render_table(log, live_lookup=live_lookup)
    stats_html = _render_stats_tab(stats or compute_bet_stats(log), panel_year=panel_year)
    labels_json = json.dumps(traj_labels)
    values_json = json.dumps(traj_values)
    odds_hist = stats or compute_bet_stats(log)
    odds_labels_json = json.dumps([b["label"] for b in odds_hist["odds_buckets"]])
    odds_pl_json = json.dumps([round(b["profit_units"], 2) for b in odds_hist["odds_buckets"]])
    odds_bets_json = json.dumps([b["bets"] for b in odds_hist["odds_buckets"]])
    stake_labels_json = json.dumps([b["label"] for b in odds_hist["stake_buckets"]])
    stake_pl_json = json.dumps([round(b["profit_units"], 2) for b in odds_hist["stake_buckets"]])
    stake_bets_json = json.dumps([b["bets"] for b in odds_hist["stake_buckets"]])
    ev_labels_json = json.dumps([b["label"] for b in odds_hist["ev_buckets"]])
    ev_pl_json = json.dumps([round(b["profit_units"], 2) for b in odds_hist["ev_buckets"]])
    ev_bets_json = json.dumps([b["bets"] for b in odds_hist["ev_buckets"]])
    model_p_labels_json = json.dumps([b["label"] for b in odds_hist["model_p_buckets"]])
    model_p_pl_json = json.dumps([round(b["profit_units"], 2) for b in odds_hist["model_p_buckets"]])
    model_p_bets_json = json.dumps([b["bets"] for b in odds_hist["model_p_buckets"]])
    n_pending = int(summary.get("n_pending", 0))
    subtitle_parts: list[str] = []
    if generated_at is not None:
        ts_gen = pd.Timestamp(generated_at).tz_convert("America/New_York")
        subtitle_parts.append(f"Updated {ts_gen.strftime('%b %d %I:%M %p ET').lstrip('0')}")
    if n_pending and live_fetched_at is not None:
        ts = pd.Timestamp(live_fetched_at).tz_convert("America/New_York")
        subtitle_parts.append(
            f"Live scores as of {ts.strftime('%I:%M %p ET').lstrip('0')} "
            f"(MLB Stats API, refreshes every 10 min)"
        )
    subtitle = " · ".join(subtitle_parts)
    subtitle_html = (
        f'<div class="updated">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    meta_refresh = ""
    if n_pending:
        meta_refresh = '<meta http-equiv="refresh" content="600">'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
{meta_refresh}
<title>MLB EV — Bet Tracker</title>
<style>
  :root {{
    --bg: #0f1115; --panel: #161a22; --border: #2a2f3a;
    --text: #e6edf3; --muted: #8b949e;
    --pos: #3fb950; --neg: #f85149; --neut: #d29922;
    --accent: #58a6ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size: 14px; line-height: 1.5; }}
  header {{ padding: 24px 32px; border-bottom: 1px solid var(--border); }}
  header h1 {{ margin: 0; font-size: 22px; font-weight: 600; }}
  header .updated {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .container {{ padding: 24px 32px; max-width: 1400px; margin: 0 auto; }}
  .cards {{ display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; }}
  .card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 8px; }}
  .card .policy {{ color: var(--muted); font-size: 11px; font-style: italic;
                   text-transform: none; letter-spacing: normal; margin: -4px 0 8px 0; }}
  .card .value {{ font-size: 24px; font-weight: 600; }}
  .card .sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .pos {{ color: var(--pos); }} .neg {{ color: var(--neg); }} .neut {{ color: var(--neut); }}
  .panel {{ background: var(--panel); border: 1px solid var(--border);
            border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  .panel h2 {{ margin: 0 0 12px 0; font-size: 16px; font-weight: 600; }}
  #chartWrap {{ height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: left;
            border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th {{ position: sticky; top: 0; background: var(--panel);
        color: var(--muted); font-weight: 500; text-transform: uppercase;
        font-size: 11px; letter-spacing: 0.04em; cursor: pointer; user-select: none; }}
  th:hover {{ color: var(--text); }}
  tr.won td {{ background-color: rgba(63, 185, 80, 0.08); }}
  tr.lost td {{ background-color: rgba(248, 81, 73, 0.08); }}
  tr.push td {{ background-color: rgba(210, 153, 34, 0.08); }}
  tr.pending td {{ opacity: 0.7; }}
  td.matchup-pending {{ white-space: normal; max-width: 320px;
                       line-height: 1.35; font-size: 12px; color: var(--muted); }}
  td.live-score {{ font-variant-numeric: tabular-nums; font-size: 12px; white-space: nowrap; }}
  td.live-score .live-good {{ color: var(--pos); font-weight: 600; }}
  td.live-score .live-bad {{ color: var(--neg); font-weight: 600; }}
  td.live-score .live-sep {{ color: var(--muted); font-weight: 400; }}
  td.live-score .live-meta {{ color: var(--muted); font-weight: 400; }}
  td.live-score.final .live-meta {{ color: var(--muted); }}
  td.live-score.pregame {{ color: var(--muted); }}
  .outcome-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                     font-size: 11px; font-weight: 600; text-transform: uppercase; }}
  .outcome-badge.won {{ background: rgba(63, 185, 80, 0.2); color: var(--pos); }}
  .outcome-badge.lost {{ background: rgba(248, 81, 73, 0.2); color: var(--neg); }}
  .outcome-badge.push {{ background: rgba(210, 153, 34, 0.2); color: var(--neut); }}
  .outcome-badge.pending {{ background: rgba(139, 148, 158, 0.2); color: var(--muted); }}
  .filters {{ display: flex; gap: 8px; margin-bottom: 12px; align-items: center; }}
  .filters input, .filters select {{ background: var(--bg); color: var(--text);
                                     border: 1px solid var(--border); border-radius: 4px;
                                     padding: 6px 10px; font-size: 13px; }}
  .scroll {{ overflow-x: auto; }}
  .empty {{ color: var(--muted); text-align: center; padding: 40px; }}
  .tabs {{ display: flex; gap: 4px; padding: 0 32px; border-bottom: 1px solid var(--border); }}
  .tab-btn {{ background: none; border: none; color: var(--muted); cursor: pointer;
              padding: 12px 18px; font-size: 14px; font-weight: 500;
              border-bottom: 2px solid transparent; margin-bottom: -1px; }}
  .tab-btn:hover {{ color: var(--text); }}
  .tab-btn.active {{ color: var(--text); border-bottom-color: var(--accent); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .split-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                  gap: 16px; margin-bottom: 24px; }}
  .side-split-card .side-pl {{ font-size: 28px; font-weight: 700; margin-top: 10px; line-height: 1.2; }}
  .side-split-card .side-roi {{ font-size: 20px; font-weight: 600; margin-top: 6px; line-height: 1.2; }}
  .stats-note {{ color: var(--muted); font-size: 12px; margin: 0 0 12px 0; }}
  .team-split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .team-split h3 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: var(--text); }}
  @media (max-width: 960px) {{ .team-split {{ grid-template-columns: 1fr; }} }}
  .chart-split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .chart-split h3 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: var(--text); }}
  .hist-wrap {{ height: 280px; }}
  @media (max-width: 960px) {{ .chart-split {{ grid-template-columns: 1fr; }} }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<header>
  <h1>MLB EV — Bet Tracker</h1>
  {subtitle_html}
</header>
<nav class="tabs" role="tablist">
  <button class="tab-btn active" type="button" role="tab" aria-selected="true"
          data-tab="bets">Bets</button>
  <button class="tab-btn" type="button" role="tab" aria-selected="false"
          data-tab="stats">Stats</button>
</nav>
<div class="container">
  {cards}
  <div id="tab-bets" class="tab-panel active" role="tabpanel">
  <div class="panel">
    <h2>Cumulative P/L — {panel_year} (Kelly-scaled units at risk)</h2>
    <div id="chartWrap"><canvas id="trajectoryChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>{panel_year} recommended bets</h2>
    <div class="filters">
      <input id="filter" type="text" placeholder="Filter teams, book, outcome…" style="flex:1;">
      <select id="outcomeFilter">
        <option value="">All outcomes</option>
        <option value="won">Won</option>
        <option value="lost">Lost</option>
        <option value="push">Push</option>
        <option value="pending">Pending</option>
      </select>
    </div>
    <div class="scroll">{table}</div>
  </div>
  </div>
  <div id="tab-stats" class="tab-panel" role="tabpanel">
  {stats_html}
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
  const chartLabels = {labels_json};
  const chartValues = {values_json};
  if (chartLabels.length) {{
    const ctx = document.getElementById('trajectoryChart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: chartLabels,
        datasets: [{{
          label: 'Cumulative P/L (u)',
          data: chartValues,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.15)',
          fill: true,
          tension: 0.2,
          pointRadius: 3,
          pointHoverRadius: 5,
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        scales: {{
          x: {{
            type: 'category',
            grid: {{ color: '#2a2f3a' }},
            ticks: {{ color: '#8b949e', maxRotation: 45, autoSkip: true, maxTicksLimit: 24 }},
          }},
          y: {{ grid: {{ color: '#2a2f3a' }}, ticks: {{ color: '#8b949e' }} }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }}
      }}
    }});
  }} else {{
    document.getElementById('chartWrap').innerHTML =
      '<div class="empty">No settled bets yet — chart will appear once games complete.</div>';
  }}

  // Tab switching
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');
  tabBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      const id = btn.dataset.tab;
      tabBtns.forEach(b => {{
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
      }});
      tabPanels.forEach(p => {{
        p.classList.toggle('active', p.id === 'tab-' + id);
      }});
      if (id === 'stats') {{
        if (window.oddsHistChart) window.oddsHistChart.resize();
        if (window.stakeHistChart) window.stakeHistChart.resize();
        if (window.evHistChart) window.evHistChart.resize();
        if (window.modelPHistChart) window.modelPHistChart.resize();
      }}
    }});
  }});

  // P/L bar charts (Stats tab)
  function makePlBarChart(canvasId, wrapId, labels, plValues, betCounts, storeKey) {{
    if (!labels.length || !document.getElementById(canvasId)) return;
    const histColors = plValues.map(v => v >= 0 ? 'rgba(63, 185, 80, 0.75)' : 'rgba(248, 81, 73, 0.75)');
    const hctx = document.getElementById(canvasId).getContext('2d');
    window[storeKey] = new Chart(hctx, {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [{{
          label: 'P/L (u)',
          data: plValues,
          backgroundColor: histColors,
          borderColor: histColors.map(c => c.replace('0.75', '1')),
          borderWidth: 1,
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        scales: {{
          x: {{
            grid: {{ color: '#2a2f3a' }},
            ticks: {{ color: '#8b949e', maxRotation: 35 }},
          }},
          y: {{
            grid: {{ color: '#2a2f3a' }},
            ticks: {{ color: '#8b949e' }},
            title: {{ display: true, text: 'Profit (units)', color: '#8b949e' }},
          }}
        }},
        plugins: {{
          legend: {{ labels: {{ color: '#e6edf3' }} }},
          tooltip: {{
            callbacks: {{
              afterLabel: (ctx) => {{
                const n = betCounts[ctx.dataIndex];
                return n + ' settled bet' + (n === 1 ? '' : 's');
              }}
            }}
          }}
        }}
      }}
    }});
  }}

  const oddsLabels = {odds_labels_json};
  const oddsPl = {odds_pl_json};
  const oddsBets = {odds_bets_json};
  makePlBarChart('oddsHistChart', 'oddsHistWrap', oddsLabels, oddsPl, oddsBets, 'oddsHistChart');
  if (!oddsLabels.length && document.getElementById('oddsHistWrap')) {{
    document.getElementById('oddsHistWrap').innerHTML =
      '<div class="empty">No settled bets yet.</div>';
  }}

  const stakeLabels = {stake_labels_json};
  const stakePl = {stake_pl_json};
  const stakeBets = {stake_bets_json};
  makePlBarChart('stakeHistChart', 'stakeHistWrap', stakeLabels, stakePl, stakeBets, 'stakeHistChart');
  if (!stakeLabels.length && document.getElementById('stakeHistWrap')) {{
    document.getElementById('stakeHistWrap').innerHTML =
      '<div class="empty">No settled bets yet.</div>';
  }}

  const evLabels = {ev_labels_json};
  const evPl = {ev_pl_json};
  const evBets = {ev_bets_json};
  makePlBarChart('evHistChart', 'evHistWrap', evLabels, evPl, evBets, 'evHistChart');
  if (!evLabels.length && document.getElementById('evHistWrap')) {{
    document.getElementById('evHistWrap').innerHTML =
      '<div class="empty">No settled bets yet.</div>';
  }}

  const modelPLabels = {model_p_labels_json};
  const modelPPl = {model_p_pl_json};
  const modelPBets = {model_p_bets_json};
  makePlBarChart('modelPHistChart', 'modelPHistWrap', modelPLabels, modelPPl, modelPBets, 'modelPHistChart');
  if (!modelPLabels.length && document.getElementById('modelPHistWrap')) {{
    document.getElementById('modelPHistWrap').innerHTML =
      '<div class="empty">No settled bets yet.</div>';
  }}

  // Shared click-to-sort for any table
  function wireSortableTable(table) {{
    if (!table) return;
    table.querySelectorAll('th').forEach((th, i) => {{
      let asc = false;
      th.addEventListener('click', () => {{
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort((a, b) => {{
          const aV = a.children[i].dataset.sort || a.children[i].textContent;
          const bV = b.children[i].dataset.sort || b.children[i].textContent;
          const aN = parseFloat(aV), bN = parseFloat(bV);
          if (!isNaN(aN) && !isNaN(bN)) return asc ? aN - bN : bN - aN;
          return asc ? aV.localeCompare(bV) : bV.localeCompare(aV);
        }});
        asc = !asc;
        rows.forEach(r => tbody.appendChild(r));
      }});
    }});
  }}

  // Bets table filtering + sorting
  const table = document.querySelector('#tab-bets table');
  if (!table) {{ /* no bets yet */ }} else {{
  const inp = document.getElementById('filter');
  const outF = document.getElementById('outcomeFilter');
  function applyFilters() {{
    const q = inp.value.toLowerCase();
    const oc = outF.value;
    table.querySelectorAll('tbody tr').forEach(tr => {{
      const text = tr.textContent.toLowerCase();
      const cls = tr.className;
      const matchText = !q || text.includes(q);
      const matchOc = !oc || cls.includes(oc);
      tr.style.display = (matchText && matchOc) ? '' : 'none';
    }});
  }}
  inp.addEventListener('input', applyFilters);
  outF.addEventListener('change', applyFilters);
  wireSortableTable(table);
  }}

  document.querySelectorAll('#tab-stats .stats-table').forEach(wireSortableTable);
</script>
</body>
</html>
"""


def _render_side_card(side: str, s: dict[str, Any]) -> str:
    title = "Home picks" if side == "home" else "Away picks"
    pl = float(s.get("profit_units", 0))
    pl_cls = "pos" if pl > 0 else ("neg" if pl < 0 else "neut")
    roi = float(s.get("roi_pct", 0))
    roi_cls = "pos" if roi > 0 else ("neg" if roi < 0 else "neut")
    avg_win_p = s.get("avg_win_p")
    prob_note = f"Avg win p {_fmt_pct(avg_win_p)}" if avg_win_p is not None else ""
    return f"""<div class="card side-split-card">
  <div class="label">{html.escape(title)}</div>
  <div class="value">{int(s.get('wins', 0))} – {int(s.get('losses', 0))}</div>
  <div class="sub">{int(s.get('bets', 0))} settled bets</div>
  <div class="side-pl {pl_cls}">{_fmt_money(pl)}</div>
  <div class="side-roi {roi_cls}">{roi:+.1f}% ROI</div>
  <div class="sub">{html.escape(prob_note)}</div>
</div>"""


def _render_team_stats_table(rows: list[dict[str, Any]], *, table_id: str) -> str:
    """Team stats table (settled bets only)."""
    if not rows:
        return '<div class="empty">No settled bets yet.</div>'
    body: list[str] = []
    for row in rows:
        pl = float(row.get("profit_units", 0))
        pl_cls = "pos" if pl > 0 else ("neg" if pl < 0 else "neut")
        roi = float(row.get("roi_pct", 0))
        record = f'{int(row.get("wins", 0))}–{int(row.get("losses", 0))}'
        team_disp = _matchup_team_display(str(row.get("team", "")))
        avg_win_p = row.get("avg_win_p")
        win_p_sort = float(avg_win_p) if avg_win_p is not None else -1.0
        hit_rate = row.get("hit_rate")
        hit_rate_sort = float(hit_rate) if hit_rate is not None else -1.0
        calib_gap_pp = row.get("calib_gap_pp")
        gap_sort = float(calib_gap_pp) if calib_gap_pp is not None else 0.0
        gap_cls = "neg" if (calib_gap_pp is not None and calib_gap_pp > 0) else (
            "pos" if (calib_gap_pp is not None and calib_gap_pp < 0) else "neut"
        )
        body.append(
            f"<tr>"
            f'<td data-sort="{html.escape(team_disp.lower())}">{html.escape(team_disp)}</td>'
            f'<td class="num" data-sort="{int(row.get("bets", 0))}">{int(row.get("bets", 0))}</td>'
            f'<td class="num" data-sort="{int(row.get("wins", 0))}">{html.escape(record)}</td>'
            f'<td class="num {pl_cls}" data-sort="{pl:.4f}">{_fmt_money(pl)}</td>'
            f'<td class="num {pl_cls}" data-sort="{roi:.4f}">{roi:+.1f}%</td>'
            f'<td class="num" data-sort="{win_p_sort:.6f}">{_fmt_pct(avg_win_p)}</td>'
            f'<td class="num" data-sort="{hit_rate_sort:.6f}">{_fmt_pct(hit_rate)}</td>'
            f'<td class="num {gap_cls}" data-sort="{gap_sort:.4f}">{_fmt_pp_gap(calib_gap_pp)}</td>'
            f"</tr>"
        )
    return (
        f'<table id="{html.escape(table_id)}" class="stats-table"><thead><tr>'
        "<th>Team</th><th>Bets</th><th>Record</th><th>P/L</th><th>ROI</th>"
        "<th>Avg win p</th><th>Hit rate</th><th>Calib gap</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _render_bucket_table(
    buckets: list[dict[str, Any]],
    *,
    table_id: str,
    first_col: str,
) -> str:
    rows: list[str] = []
    for bucket in buckets:
        pl = float(bucket.get("profit_units", 0))
        pl_cls = "pos" if pl > 0 else ("neg" if pl < 0 else "neut")
        roi = float(bucket.get("roi_pct", 0))
        rows.append(
            f"<tr>"
            f"<td>{html.escape(str(bucket.get('label', '')))}</td>"
            f'<td class="num" data-sort="{int(bucket.get("bets", 0))}">'
            f'{int(bucket.get("bets", 0))}</td>'
            f'<td class="num" data-sort="{int(bucket.get("wins", 0))}">'
            f'{int(bucket.get("wins", 0))}–{int(bucket.get("losses", 0))}</td>'
            f'<td class="num {pl_cls}" data-sort="{pl:.4f}">{_fmt_money(pl)}</td>'
            f'<td class="num {pl_cls}" data-sort="{roi:.4f}">{roi:+.1f}%</td>'
            f"</tr>"
        )
    if not rows:
        return ""
    return (
        f'<table id="{html.escape(table_id)}" class="stats-table"><thead><tr>'
        f"<th>{html.escape(first_col)}</th><th>Bets</th><th>Record</th><th>P/L</th><th>ROI</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _render_stats_tab(stats: dict[str, Any], *, panel_year: str) -> str:
    sides = stats.get("sides", {})
    home = sides.get("home", {})
    away = sides.get("away", {})
    split_cards = (
        f'<div class="split-cards">{_render_side_card("home", home)}'
        f'{_render_side_card("away", away)}</div>'
    )

    odds_table = _render_bucket_table(
        stats.get("odds_buckets", []),
        table_id="oddsBucketTable",
        first_col="Odds bucket",
    )
    stake_table = _render_bucket_table(
        stats.get("stake_buckets", []),
        table_id="stakeBucketTable",
        first_col="Bet size",
    )
    ev_table = _render_bucket_table(
        stats.get("ev_buckets", []),
        table_id="evBucketTable",
        first_col="Predicted EV",
    )
    model_p_table = _render_bucket_table(
        stats.get("model_p_buckets", []),
        table_id="modelPBucketTable",
        first_col="Model win p",
    )

    team_for_table = _render_team_stats_table(
        stats.get("teams_for", []), table_id="teamStatsForTable",
    )
    team_against_table = _render_team_stats_table(
        stats.get("teams_against", []), table_id="teamStatsAgainstTable",
    )
    if not stats.get("teams_for") and not stats.get("teams_against"):
        team_split = '<div class="empty">No bets logged yet.</div>'
    else:
        team_split = f"""
    <div class="team-split">
      <div>
        <h3>Teams picked</h3>
        <div class="scroll">{team_for_table}</div>
      </div>
      <div>
        <h3>Teams faded</h3>
        <div class="scroll">{team_against_table}</div>
      </div>
    </div>"""

    return f"""
  <p class="stats-note">Settled-bet P/L and ROI for {html.escape(panel_year)}. Refreshes on every odds pull.</p>
  {split_cards}
  <div class="panel">
    <h2>P/L breakdown — {html.escape(panel_year)}</h2>
    <p class="stats-note">Settled bets only.</p>
    <div class="chart-split">
      <div>
        <h3>By odds</h3>
        <p class="stats-note">American odds at recommendation time.</p>
        <div class="hist-wrap" id="oddsHistWrap"><canvas id="oddsHistChart"></canvas></div>
        <div class="scroll" style="margin-top:16px">{odds_table}</div>
      </div>
      <div>
        <h3>By bet size</h3>
        <p class="stats-note">Kelly-scaled units at risk (0.5–2u).</p>
        <div class="hist-wrap" id="stakeHistWrap"><canvas id="stakeHistChart"></canvas></div>
        <div class="scroll" style="margin-top:16px">{stake_table}</div>
      </div>
    </div>
    <div style="margin-top:24px">
      <h3>By predicted EV</h3>
      <p class="stats-note">Model-predicted EV at recommendation time (1% bins).</p>
      <div class="hist-wrap" id="evHistWrap"><canvas id="evHistChart"></canvas></div>
      <div class="scroll" style="margin-top:16px">{ev_table}</div>
    </div>
    <div style="margin-top:24px">
      <h3>By model win probability</h3>
      <p class="stats-note">Model win probability at recommendation time.</p>
      <div class="hist-wrap" id="modelPHistWrap"><canvas id="modelPHistChart"></canvas></div>
      <div class="scroll" style="margin-top:16px">{model_p_table}</div>
    </div>
  </div>
  <div class="panel">
    <h2>Team record &amp; P/L — {html.escape(panel_year)}</h2>
    <p class="stats-note">Calib gap = avg model win p minus hit rate. Positive means the model was overconfident on that team.</p>
    {team_split}
  </div>
"""


def _render_summary_cards(s: dict) -> str:
    if not s or s.get("n_bets", 0) == 0:
        return '<div class="empty">No bets logged yet.</div>'

    profit = float(s.get("profit_units", 0))
    profit_cls = "pos" if profit > 0 else ("neg" if profit < 0 else "neut")
    roi = float(s.get("roi_per_unit", 0)) * 100
    roi_cls = "pos" if roi > 0 else ("neg" if roi < 0 else "neut")
    avg_clv = s.get("avg_clv_pp", float("nan"))
    clv_cls = "pos" if (pd.notna(avg_clv) and avg_clv > 0) else (
              "neg" if (pd.notna(avg_clv) and avg_clv < 0) else "neut")
    clv_beat = s.get("clv_beat_rate", float("nan"))
    hit_rate = float(s.get("hit_rate", 0)) * 100

    def card(label: str, value: str, sub: str = "", cls: str = "",
             policy_note: str = "") -> str:
        note_html = f'<div class="policy">{html.escape(policy_note)}</div>' if policy_note else ""
        return (f'<div class="card"><div class="label">{label}</div>{note_html}'
                f'<div class="value {cls}">{value}</div>'
                f'<div class="sub">{sub}</div></div>')

    n_settled = s.get("n_settled", 0)
    n_pending = s.get("n_pending", 0)
    return f"""<div class="cards">
  {card("Bets logged", f'{s["n_bets"]:,}', f'{n_settled} settled · {n_pending} pending',
        policy_note=PAPER_LINEUP_POLICY_NOTE)}
  {card("Record", f'{s["n_wins"]} – {s["n_losses"]}', f'{hit_rate:.1f}% hit rate')}
  {card("Profit", _fmt_money(profit), 'Kelly-scaled units (see Risk column)', profit_cls)}
  {card("ROI / bet", f'{roi:+.2f}%', f'over {n_settled} settled bets', roi_cls)}
  {card("Avg EV at rec", f'{s.get("avg_ev_at_rec", 0) * 100:+.2f}%', 'model-predicted EV at rec time')}
  {card("Avg CLV", _fmt_pp(avg_clv) if pd.notna(avg_clv) else '—',
         (f'{clv_beat*100:.1f}% of bets beat the close' if pd.notna(clv_beat) else 'awaiting closing lines'), clv_cls)}
</div>"""


def _render_table(log: pd.DataFrame, *, live_lookup: dict | None = None) -> str:
    if log.empty:
        return '<div class="empty">No bets in the log yet. Run <code>make project</code> to populate.</div>'
    log = log.copy()
    log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
    log = log.sort_values("commence_time", ascending=False)
    pitchers_lookup = _pitcher_lookup_from_schedule_snapshots(log)
    live_lookup = live_lookup or {}
    headers = ["Date", "Matchup", "Live", "Pick", "Book", "Risk (u)", "Odds",
               "Model p", "Fair p", "Edge", "EV", "Closing", "CLV",
               "Result", "P/L"]
    th_html = "".join(f"<th>{h}</th>" for h in headers)
    rows: list[str] = []
    for _, r in log.iterrows():
        ct = pd.Timestamp(r["commence_time"]).tz_convert("America/New_York")
        date_str = ct.strftime("%a %b %d %H:%M ET")
        pick = html.escape(str(r["recommended_team"]))
        book = html.escape(str(r["book"]) if pd.notna(r["book"]) else "—")
        outcome = str(r["outcome"]) if pd.notna(r["outcome"]) else "pending"
        oc_cls = _outcome_class(outcome)
        pl = r.get("profit_units")
        if pd.notna(pl):
            pl_str = _fmt_money(float(pl))
            pl_cls = "pos" if pl > 0 else ("neg" if pl < 0 else "neut")
        else:
            pl_str = "—"; pl_cls = ""
        clv = r.get("clv_pp")
        clv_str = _fmt_pp(clv)
        clv_cls = "pos" if (pd.notna(clv) and clv > 0) else (
                  "neg" if (pd.notna(clv) and clv < 0) else "")
        gid = int(r["game_id"])
        ru = float(r["risk_units"]) if pd.notna(r.get("risk_units")) else 1.0

        aa = hh = ""
        if oc_cls == "pending":
            tup = pitchers_lookup.get(gid, ("", ""))
            aa, hh = tup[0], tup[1]

        away_raw = str(r["away_name"]).strip()
        home_raw = str(r["home_name"]).strip()
        away_disp = _matchup_team_display(away_raw)
        home_disp = _matchup_team_display(home_raw)
        if oc_cls == "pending":
            pa = _pitcher_surname_display(aa)
            ph = _pitcher_surname_display(hh)
            sort_mu = (
                f"{away_disp.lower()} ({pa.lower()}) @ "
                f"{home_disp.lower()} ({ph.lower()})"
            )
            matchup = (
                f"{html.escape(away_disp)} ({html.escape(pa)}) @ "
                f"{html.escape(home_disp)} ({html.escape(ph)})"
            )
            matchup_td = (
                f'<td class="matchup-pending" data-sort="{html.escape(sort_mu)}">'
                f"{matchup}</td>"
            )
        else:
            sort_mu = f"{away_disp} @ {home_disp}".lower()
            matchup = f"{html.escape(away_disp)} @ {html.escape(home_disp)}"
            matchup_td = f'<td data-sort="{html.escape(sort_mu)}">{matchup}</td>'

        if oc_cls == "pending":
            from src.tracking.live_scores import _inning_label

            state = live_lookup.get(gid)
            live_html, live_tip, live_cls = format_live_score(
                state,
                recommended_team=str(r["recommended_team"]),
                away_name=away_raw,
                home_name=home_raw,
            )
            if state is not None and live_cls in ("live", "final"):
                away_s = state.away_score if state.away_score is not None else 0
                home_s = state.home_score if state.home_score is not None else 0
                live_sort = f"{away_s}-{home_s} {_inning_label(state).lower()}".strip()
            else:
                live_sort = live_tip.lower()
            live_td = (
                f'<td class="live-score {live_cls}" data-sort="{html.escape(live_sort)}" '
                f'title="{html.escape(live_tip)}">{live_html}</td>'
            )
        else:
            live_td = '<td class="live-score pregame" data-sort="">—</td>'

        rows.append(
            f'<tr class="{oc_cls}">'
            f'<td data-sort="{ct.isoformat()}">{date_str}</td>'
            f'{matchup_td}'
            f'{live_td}'
            f'<td><strong>{pick}</strong></td>'
            f'<td>{book}</td>'
            f'<td data-sort="{ru:.4f}">{ru:.2f}</td>'
            f'<td data-sort="{float(r["odds_at_rec"])}">{_fmt_american(r["odds_at_rec"])}</td>'
            f'<td data-sort="{float(r["model_p"])}">{_fmt_pct(r["model_p"])}</td>'
            f'<td data-sort="{float(r["fair_p_at_rec"])}">{_fmt_pct(r["fair_p_at_rec"])}</td>'
            f'<td data-sort="{float(r["edge_at_rec"]) * 100}">{r["edge_at_rec"] * 100:+.1f}pp</td>'
            f'<td data-sort="{float(r["ev_at_rec"]) * 100}">{r["ev_at_rec"] * 100:+.2f}%</td>'
            f'<td>{_fmt_american(r.get("closing_odds"))}</td>'
            f'<td data-sort="{float(clv) if pd.notna(clv) else -999}" class="{clv_cls}">{clv_str}</td>'
            f'<td><span class="outcome-badge {oc_cls}">{outcome}</span></td>'
            f'<td class="{pl_cls}" data-sort="{float(pl) if pd.notna(pl) else -999}">{pl_str}</td>'
            f'</tr>'
        )
    return f"<table><thead><tr>{th_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = render()
    print(f"Wrote {p}")
