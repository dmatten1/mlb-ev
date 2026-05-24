"""Live game state for pending bets via MLB-StatsAPI (free, no API key).

Used by :mod:`src.tracking.dashboard` to show in-progress scores in the
bet table. One ``schedule`` call per distinct game date covers all pending
``game_id`` rows for that day.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

import pandas as pd

logger = logging.getLogger("tracking.live_scores")


@dataclass(frozen=True)
class LiveGameState:
    game_id: int
    away_score: int | None
    home_score: int | None
    abstract_status: str | None
    detailed_status: str | None
    inning: int | None
    inning_ordinal: str | None
    inning_state: str | None
    game_datetime: str | None


def _safe_int(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _state_from_schedule_game(g: dict) -> LiveGameState:
    teams = g.get("teams") or {}
    home_t = teams.get("home") or {}
    away_t = teams.get("away") or {}
    status = g.get("status") or {}
    linescore = g.get("linescore") or {}
    return LiveGameState(
        game_id=int(g["gamePk"]),
        away_score=_safe_int(away_t.get("score")),
        home_score=_safe_int(home_t.get("score")),
        abstract_status=status.get("abstractGameState"),
        detailed_status=status.get("detailedState"),
        inning=_safe_int(linescore.get("currentInning")),
        inning_ordinal=linescore.get("currentInningOrdinal"),
        inning_state=linescore.get("inningState"),
        game_datetime=g.get("gameDate"),
    )


def _fetch_game_by_pk(game_id: int) -> LiveGameState | None:
    """Fetch one game's live state by ``gamePk`` (works across official dates)."""
    import statsapi

    try:
        raw = statsapi.get("game", {"gamePk": int(game_id), "hydrate": "linescore"})
    except Exception as e:  # noqa: BLE001
        logger.warning("live scores game fetch failed for %s: %s", game_id, e)
        return None

    live = raw.get("liveData") or {}
    linescore = live.get("linescore") or {}
    status = live.get("status") or (raw.get("gameData") or {}).get("status") or {}
    teams = linescore.get("teams") or {}
    away_t = teams.get("away") or {}
    home_t = teams.get("home") or {}
    game_data = raw.get("gameData") or {}
    datetime_info = (game_data.get("datetime") or {}).get("dateTime")
    return LiveGameState(
        game_id=int(game_id),
        away_score=_safe_int(away_t.get("runs", away_t.get("score"))),
        home_score=_safe_int(home_t.get("runs", home_t.get("score"))),
        abstract_status=status.get("abstractGameState"),
        detailed_status=status.get("detailedState"),
        inning=_safe_int(linescore.get("currentInning")),
        inning_ordinal=linescore.get("currentInningOrdinal"),
        inning_state=linescore.get("inningState"),
        game_datetime=datetime_info,
    )


def fetch_live_scores_for_game_ids(
    game_ids: Iterable[int],
    dates: Iterable[date],
) -> dict[int, LiveGameState]:
    """Return ``game_id`` → :class:`LiveGameState` for games on the given dates."""
    wanted = {int(g) for g in game_ids}
    if not wanted:
        return {}

    import statsapi

    out: dict[int, LiveGameState] = {}
    scan_dates = set(dates)
    scan_dates.add(date.today())

    for d in sorted(scan_dates):
        try:
            raw = statsapi.get("schedule", {
                "sportId": 1,
                "date": d.isoformat(),
                "hydrate": "linescore",
            })
        except Exception as e:  # noqa: BLE001
            logger.warning("live scores schedule fetch failed for %s: %s", d, e)
            continue

        for block in raw.get("dates") or []:
            for g in block.get("games") or []:
                pk = g.get("gamePk")
                if pk is None or int(pk) not in wanted:
                    continue
                out[int(pk)] = _state_from_schedule_game(g)

    missing = wanted - set(out)
    for pk in sorted(missing):
        state = _fetch_game_by_pk(pk)
        if state is not None:
            out[pk] = state
    return out


def live_scores_for_pending_log(log: pd.DataFrame) -> tuple[dict[int, LiveGameState], datetime | None]:
    """Fetch live states for rows with ``outcome == 'pending'``."""
    if log.empty or "game_id" not in log.columns:
        return {}, None

    pend = log[log["outcome"].astype(str) == "pending"]
    if pend.empty:
        return {}, None

    dates: set[date] = set()
    for gd in pend.get("game_date", pd.Series(dtype=object)):
        try:
            dates.add(pd.Timestamp(gd).normalize().date())
        except Exception:
            continue
    if not dates and "commence_time" in pend.columns:
        for ct in pend["commence_time"]:
            try:
                dates.add(pd.Timestamp(ct).tz_convert("America/New_York").date())
            except Exception:
                continue

    game_ids = pend["game_id"].dropna().astype(int).tolist()
    states = fetch_live_scores_for_game_ids(game_ids, dates)
    return states, datetime.now(timezone.utc)


def _inning_number(state: LiveGameState) -> int | None:
    if state.inning is not None:
        return state.inning
    if state.inning_ordinal:
        m = re.match(r"(\d+)", str(state.inning_ordinal))
        if m:
            return int(m.group(1))
    return None


def _inning_label(state: LiveGameState) -> str:
    """Compact, consistent inning text: ``Top 2``, ``Mid 1``, ``Bot 1``, ``End 3``."""
    n = _inning_number(state)
    if n is None:
        return ""
    st = (state.inning_state or "").strip()
    if st == "Top":
        return f"Top {n}"
    if st == "Bottom":
        return f"Bot {n}"
    if st == "Middle":
        return f"Mid {n}"
    if st == "End":
        return f"End {n}"
    if st:
        return f"{st} {n}"
    return str(n)


def _pick_side(
    recommended_team: str,
    away_name: str,
    home_name: str,
) -> str | None:
    rec = recommended_team.strip().casefold()
    away = away_name.strip().casefold()
    home = home_name.strip().casefold()
    if not rec:
        return None
    if rec == away or rec in away or away in rec:
        return "away"
    if rec == home or rec in home or home in rec:
        return "home"
    return None


def _score_html(
    away: int,
    home: int,
    *,
    pick_side: str | None,
) -> str:
    """Away–home order; green = picked team, red = opponent (score-independent)."""
    if pick_side is None:
        return f"{away}–{home}"

    if pick_side == "away":
        away_cls, home_cls = "live-good", "live-bad"
    else:
        away_cls, home_cls = "live-bad", "live-good"

    return (
        f'<span class="{away_cls}">{away}</span>'
        f'<span class="live-sep">–</span>'
        f'<span class="{home_cls}">{home}</span>'
    )


def format_live_score(
    state: LiveGameState | None,
    *,
    recommended_team: str = "",
    away_name: str = "",
    home_name: str = "",
) -> tuple[str, str, str]:
    """Return ``(html_inner, tooltip, css_class)`` for a table cell."""
    if state is None:
        return "—", "Score unavailable", "pregame"

    abstract = (state.abstract_status or "").strip()
    detailed = (state.detailed_status or "").strip()
    pick_side = _pick_side(recommended_team, away_name, home_name)

    if "Postponed" in detailed or "Cancelled" in detailed:
        return "PPD", detailed, "pregame"
    if "Delayed" in detailed and abstract not in ("Live", "Final"):
        return "Delay", detailed, "pregame"
    if abstract in ("Preview", "Pre-Game", "") or detailed in ("Scheduled", "Pre-Game", "Warmup"):
        tip = detailed or "Scheduled"
        if state.game_datetime:
            try:
                ts = pd.Timestamp(state.game_datetime).tz_convert("America/New_York")
                tip = f"First pitch {ts.strftime('%-I:%M %p ET')}"
            except Exception:
                pass
        return "—", tip, "pregame"

    away = state.away_score if state.away_score is not None else 0
    home = state.home_score if state.home_score is not None else 0
    score_html = _score_html(away, home, pick_side=pick_side)
    plain_score = f"{away}–{home}"

    if abstract == "Final" or detailed == "Final":
        tip = f"Final · {plain_score}"
        return score_html, tip, "final"

    inn = _inning_label(state)
    if inn:
        display = f'{score_html}<span class="live-meta"> · {html.escape(inn)}</span>'
        tip = f"{plain_score} · {detailed or inn}"
    else:
        display = f'{score_html}<span class="live-meta"> · Live</span>'
        tip = detailed or "Live"
    return display, tip, "live"
