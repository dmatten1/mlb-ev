"""Counterfactual dashboard: bets sized at the closing line only.

Parallel to :mod:`src.tracking.dashboard` — does not read or write
``bet_log.parquet``.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import pandas as pd

from src.tracking.bet_log import filter_log_by_season, summarize_frame
from src.tracking.close_line_tracker import build_close_line_log
from src.tracking.dashboard import (
    DEFAULT_DASHBOARD_SEASON_YEAR,
    _fmt_american,
    _fmt_money,
    _fmt_pct,
    _matchup_team_display,
    _outcome_class,
    _render_summary_cards,
)

logger = logging.getLogger("tracking.close_line_dashboard")

DEFAULT_OUT = Path("data/tracking/close_line_dashboard.html")
PAPER_DASHBOARD_HREF = "index.html"
COUNTERFACTUAL_S3_KEY = "counterfactual.html"


def render(
    out_path: Path | str = DEFAULT_OUT,
    *,
    season_year: int | None = DEFAULT_DASHBOARD_SEASON_YEAR,
    predictions_root: Path | str = "data/predictions",
) -> Path:
    log = build_close_line_log(
        predictions_root=predictions_root,
        season_year=season_year,
    )
    log = filter_log_by_season(log, season_year)
    summary = summarize_frame(log)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not log.empty:
        log = log.copy()
        log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
        log = log.sort_values("commence_time")
        settled = log[log["outcome"].isin(["won", "lost", "push"])].copy()
        settled["profit_units"] = settled["profit_units"].astype(float).fillna(0.0)
        settled["cum_profit"] = settled["profit_units"].cumsum()
        labels: list[str] = []
        for ct in settled["commence_time"]:
            ts = pd.Timestamp(ct).tz_convert("America/New_York")
            labels.append(
                ts.strftime("%b %d · ") + str(ts.hour).zfill(2) + ":" + str(ts.minute).zfill(2)
            )
        values = [float(x) for x in settled["cum_profit"]]
    else:
        labels, values = [], []

    html_doc = _build_html(log, summary, labels, values, season_year=season_year)
    out_path.write_text(html_doc, encoding="utf-8")
    logger.info("Wrote close-line dashboard to %s", out_path)
    return out_path


def _build_html(
    log: pd.DataFrame,
    summary: dict,
    traj_labels: list[str],
    traj_values: list[float],
    *,
    season_year: int | None,
) -> str:
    panel_year = str(int(season_year)) if season_year is not None else "All seasons"
    cards = _render_summary_cards(summary)
    table = _render_table(log)
    labels_json = json.dumps(traj_labels)
    values_json = json.dumps(traj_values)
    nav = (
        f'<p class="nav"><a href="{html.escape(PAPER_DASHBOARD_HREF)}">'
        f"← Paper-trading dashboard</a></p>"
    )
    expl = (
        "Counterfactual track: same model probabilities from prediction slates "
        "(projected lineups at predict time), but bet only when +EV vs the "
        "<strong>closing moneyline</strong> (latest pre-first-pitch snapshot). "
        "Does not change your paper log."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MLB EV — Closing Line Counterfactual</title>
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
  header .expl {{ color: var(--muted); font-size: 13px; margin-top: 8px; max-width: 720px; }}
  header .nav {{ margin: 8px 0 0 0; font-size: 13px; }}
  header .nav a {{ color: var(--accent); text-decoration: none; }}
  header .nav a:hover {{ text-decoration: underline; }}
  .container {{ padding: 24px 32px; max-width: 1400px; margin: 0 auto; }}
  .cards {{ display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; }}
  .card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 8px; }}
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
  tr.won td {{ background-color: rgba(63, 185, 80, 0.08); }}
  tr.lost td {{ background-color: rgba(248, 81, 73, 0.08); }}
  tr.push td {{ background-color: rgba(210, 153, 34, 0.08); }}
  tr.pending td {{ opacity: 0.7; }}
  .outcome-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                     font-size: 11px; font-weight: 600; text-transform: uppercase; }}
  .outcome-badge.won {{ background: rgba(63, 185, 80, 0.2); color: var(--pos); }}
  .outcome-badge.lost {{ background: rgba(248, 81, 73, 0.2); color: var(--neg); }}
  .outcome-badge.pending {{ background: rgba(139, 148, 158, 0.2); color: var(--muted); }}
  .filters {{ display: flex; gap: 8px; margin-bottom: 12px; }}
  .filters input, .filters select {{ background: var(--bg); color: var(--text);
                                     border: 1px solid var(--border); border-radius: 4px;
                                     padding: 6px 10px; font-size: 13px; flex: 1; }}
  .scroll {{ overflow-x: auto; }}
  .empty {{ color: var(--muted); text-align: center; padding: 40px; }}
</style>
</head>
<body>
<header>
  {nav}
  <h1>MLB EV — Closing Line (Counterfactual)</h1>
  <div class="expl">{expl}</div>
</header>
<div class="container">
  {cards}
  <div class="panel">
    <h2>Cumulative P/L — {panel_year} (close-line entry, Kelly-scaled)</h2>
    <div id="chartWrap"><canvas id="trajectoryChart"></canvas></div>
  </div>
  <div class="panel">
    <h2>{panel_year} close-line bets</h2>
    <div class="filters">
      <input id="filter" type="text" placeholder="Filter teams, book, outcome…">
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
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        scales: {{
          x: {{ type: 'category', grid: {{ color: '#2a2f3a' }},
               ticks: {{ color: '#8b949e', maxRotation: 45 }} }},
          y: {{ grid: {{ color: '#2a2f3a' }}, ticks: {{ color: '#8b949e' }} }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#e6edf3' }} }} }}
      }}
    }});
  }} else {{
    document.getElementById('chartWrap').innerHTML =
      '<div class="empty">No close-line bets yet — need prediction slates + closing odds.</div>';
  }}
  const table = document.querySelector('table');
  const inp = document.getElementById('filter');
  const outF = document.getElementById('outcomeFilter');
  function applyFilters() {{
    const q = inp.value.toLowerCase();
    const oc = outF.value;
    table.querySelectorAll('tbody tr').forEach(tr => {{
      const text = tr.textContent.toLowerCase();
      const cls = tr.className;
      tr.style.display = ((!q || text.includes(q)) && (!oc || cls.includes(oc))) ? '' : 'none';
    }});
  }}
  inp.addEventListener('input', applyFilters);
  outF.addEventListener('change', applyFilters);
</script>
</body>
</html>
"""


def _render_table(log: pd.DataFrame) -> str:
    if log.empty:
        return (
            '<div class="empty">No close-line +EV bets found for this season yet. '
            "Requires saved prediction parquets and closing odds snapshots.</div>"
        )
    log = log.copy()
    log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
    log = log.sort_values("commence_time", ascending=False)
    headers = [
        "Date", "Matchup", "Pick", "Book", "Risk (u)", "Close odds",
        "Model p", "Fair p (close)", "Edge", "EV", "Lineups", "Result", "P/L",
    ]
    th_html = "".join(f"<th>{h}</th>" for h in headers)
    rows: list[str] = []
    for _, r in log.iterrows():
        ct = pd.Timestamp(r["commence_time"]).tz_convert("America/New_York")
        date_str = ct.strftime("%a %b %d %H:%M ET")
        away_disp = _matchup_team_display(str(r["away_name"]))
        home_disp = _matchup_team_display(str(r["home_name"]))
        matchup = f"{html.escape(away_disp)} @ {html.escape(home_disp)}"
        pick = html.escape(str(r["recommended_team"]))
        book = html.escape(str(r["book"]) if pd.notna(r["book"]) else "—")
        outcome = str(r["outcome"]) if pd.notna(r["outcome"]) else "pending"
        oc_cls = _outcome_class(outcome)
        pl = r.get("profit_units")
        pl_str = _fmt_money(float(pl)) if pd.notna(pl) else "—"
        pl_cls = "pos" if pd.notna(pl) and pl > 0 else ("neg" if pd.notna(pl) and pl < 0 else "")
        ru = float(r["risk_units"]) if pd.notna(r.get("risk_units")) else 1.0
        lh = str(r.get("lineup_source_home") or "P")[0].upper()
        la = str(r.get("lineup_source_away") or "P")[0].upper()
        rows.append(
            f'<tr class="{oc_cls}">'
            f'<td data-sort="{ct.isoformat()}">{date_str}</td>'
            f'<td data-sort="{matchup.lower()}">{matchup}</td>'
            f'<td><strong>{pick}</strong></td>'
            f'<td>{book}</td>'
            f'<td data-sort="{ru:.4f}">{ru:.2f}</td>'
            f'<td data-sort="{float(r["odds_at_rec"])}">{_fmt_american(r["odds_at_rec"])}</td>'
            f'<td data-sort="{float(r["model_p"])}">{_fmt_pct(r["model_p"])}</td>'
            f'<td data-sort="{float(r["fair_p_at_rec"])}">{_fmt_pct(r["fair_p_at_rec"])}</td>'
            f'<td data-sort="{float(r["edge_at_rec"]) * 100}">{r["edge_at_rec"] * 100:+.1f}pp</td>'
            f'<td data-sort="{float(r["ev_at_rec"]) * 100}">{r["ev_at_rec"] * 100:+.2f}%</td>'
            f'<td>{lh}/{la}</td>'
            f'<td><span class="outcome-badge {oc_cls}">{outcome}</span></td>'
            f'<td class="{pl_cls}">{pl_str}</td>'
            f'</tr>'
        )
    return f"<table><thead><tr>{th_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = render()
    print(f"Wrote {p}")
