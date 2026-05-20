"""Static HTML dashboard rendered from the bet log.

Renders a single self-contained ``bet_dashboard.html`` with three sections:

1. **Summary cards** — overall stats (bets settled, wins, profit in
   units, ROI per bet, hit rate, average CLV, CLV beat rate).
2. **Bankroll trajectory** — cumulative profit-per-unit over time, with
   per-day markers (Chart.js, fed inline as a JSON array).
3. **Bet table** — every recommended bet with date,
   matchup (pending rows embed probable SP, e.g. ``Brewers (Brown) @ Cubs (Smith)``),
   book, recommended side, model p, fair p at rec,
   CLV (pp), outcome, P/L.
   Sortable + filterable in-browser via a tiny vanilla JS script.

No external CSS/JS files: Chart.js is loaded from a CDN inside the
HTML. Open the file in any browser.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.tracking.bet_log import (
    DEFAULT_LOG_PATH,
    filter_log_by_season,
    load_log,
    summarize_frame,
)

logger = logging.getLogger("tracking.dashboard")


DEFAULT_OUT = Path("data/tracking/bet_dashboard.html")

# Hold-out / live test year: summary + chart + table use Kelly-scaled rows for
# this UTC commence year only. Set to ``None`` to include every season in the log.
DEFAULT_DASHBOARD_SEASON_YEAR: int | None = 2026


def _fmt_money(units: float) -> str:
    sign = "+" if units >= 0 else "−"
    return f"{sign}{abs(units):.2f}u"


def _fmt_pct(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "—"
    return f"{float(p) * 100:.1f}%"


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
            "pending": "pending"}.get(str(outcome), "")


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


def _pitcher_lookup_from_schedule_snapshots(
    log: pd.DataFrame,
) -> dict[int, tuple[str, str]]:
    """``game_id`` → (away probable SP, home probable SP) from local schedule JSON.

    Rows are read from ``data/raw/schedule/...`` written by
    :func:`src.ingest.fetch_schedule.load_schedule_for_date`. Used for **pending**
    bets only; shows StatsAPI *probable* starter names for that snapshot.
    """
    lookup: dict[int, tuple[str, str]] = {}
    if log.empty or "game_date" not in log.columns:
        return lookup

    pend = log[log["outcome"].astype(str) == "pending"]
    if pend.empty:
        return lookup

    from src.ingest.fetch_schedule import load_schedule_for_date

    seen_dates: set = set()
    for gd in pend["game_date"]:
        try:
            seen_dates.add(pd.Timestamp(gd).normalize().date())
        except Exception:
            continue

    for d in sorted(seen_dates):
        sdf = load_schedule_for_date(d)
        if sdf.empty:
            continue
        if ("away_probable_pitcher_name" not in sdf.columns
                or "home_probable_pitcher_name" not in sdf.columns):
            continue
        for _, row in sdf.iterrows():
            try:
                gid = int(row["game_id"])
            except (TypeError, ValueError, KeyError):
                continue
            ap = row.get("away_probable_pitcher_name")
            hp = row.get("home_probable_pitcher_name")
            aa = str(ap).strip() if pd.notna(ap) and ap is not None else ""
            hh = str(hp).strip() if pd.notna(hp) and hp is not None else ""
            lookup[gid] = (aa, hh)
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
    log = filter_log_by_season(full, season_year)
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

    html_doc = _build_html(log, summary, labels, values, season_year=season_year)
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
) -> str:
    generated = datetime.now().isoformat(timespec="seconds")
    if season_year is not None:
        scope_note = f"{int(season_year)} season (UTC commence year) · Kelly-scaled risk"
    else:
        scope_note = "All seasons · Kelly-scaled risk"
    panel_year = str(int(season_year)) if season_year is not None else "All seasons"
    cards = _render_summary_cards(summary)
    table = _render_table(log)
    labels_json = json.dumps(traj_labels)
    values_json = json.dumps(traj_values)
    n_pending = summary.get("n_pending", 0)
    scope_escaped = html.escape(scope_note)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
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
  header .gen {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
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
  th:hover {{ color: var(--text); }}
  tr.won td {{ background-color: rgba(63, 185, 80, 0.08); }}
  tr.lost td {{ background-color: rgba(248, 81, 73, 0.08); }}
  tr.push td {{ background-color: rgba(210, 153, 34, 0.08); }}
  tr.pending td {{ opacity: 0.7; }}
  td.matchup-pending {{ white-space: normal; max-width: 320px;
                       line-height: 1.35; font-size: 12px; color: var(--muted); }}
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
</style>
</head>
<body>
<header>
  <h1>MLB EV — Bet Tracker</h1>
  <div class="gen">Generated {generated} · {scope_escaped} · {n_pending} bets pending settlement</div>
</header>
<div class="container">
  {cards}
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
  // Table filtering + sorting
  const table = document.querySelector('table');
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
  // Click-to-sort
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
</script>
</body>
</html>
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

    def card(label: str, value: str, sub: str = "", cls: str = "") -> str:
        return (f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value {cls}">{value}</div>'
                f'<div class="sub">{sub}</div></div>')

    n_settled = s.get("n_settled", 0)
    n_pending = s.get("n_pending", 0)
    return f"""<div class="cards">
  {card("Bets logged", f'{s["n_bets"]:,}', f'{n_settled} settled · {n_pending} pending')}
  {card("Record", f'{s["n_wins"]} – {s["n_losses"]}', f'{hit_rate:.1f}% hit rate')}
  {card("Profit", _fmt_money(profit), 'Kelly-scaled units (see Risk column)', profit_cls)}
  {card("ROI / bet", f'{roi:+.2f}%', f'over {n_settled} settled bets', roi_cls)}
  {card("Avg EV at rec", f'{s.get("avg_ev_at_rec", 0) * 100:+.2f}%', 'model-predicted EV at rec time')}
  {card("Avg CLV", _fmt_pp(avg_clv) if pd.notna(avg_clv) else '—',
         (f'{clv_beat*100:.1f}% of bets beat the close' if pd.notna(clv_beat) else 'awaiting closing lines'), clv_cls)}
</div>"""


def _render_table(log: pd.DataFrame) -> str:
    if log.empty:
        return '<div class="empty">No bets in the log yet. Run <code>make project</code> to populate.</div>'
    log = log.copy()
    log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
    log = log.sort_values("commence_time", ascending=False)
    pitchers_lookup = _pitcher_lookup_from_schedule_snapshots(log)
    headers = ["Date", "Matchup", "Pick", "Book", "Risk (u)", "Odds",
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

        rows.append(
            f'<tr class="{oc_cls}">'
            f'<td data-sort="{ct.isoformat()}">{date_str}</td>'
            f'{matchup_td}'
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
