"""Adaptive blend of production vs team-RS probabilities for tail teams.

Uses matchup tail flags (elite / cellar involvement) with weights fit on a
held-out calibration slice. Middle-tier games stay closer to production;
games involving best/worst teams lean on team RS (capped to avoid fav overshoot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-9


@dataclass(frozen=True)
class TailBlendModel:
    """Fitted tail-aware probability blender."""

    league_avg_runs: float
    league_avg_raw_runs: float
    elite_margin: float
    cellar_margin: float
    weight_elite: float
    weight_cellar: float
    weight_both_tail: float
    weight_middle: float
    fav_cap_weight: float
    fav_threshold: float = 0.58
    min_elite_home_weight: float = 0.50
    min_cellar_home_weight: float = 0.55
    platt: LogisticRegression | None = None
    version: str = "v1-tail-blend"

    def blend_row(
        self,
        p_prod: float,
        p_team: float,
        home_base: float,
        away_base: float,
    ) -> float:
        flags = tail_flags(
            home_base, away_base, self.league_avg_runs,
            elite_margin=self.elite_margin,
            cellar_margin=self.cellar_margin,
        )
        w = self._category_weight(flags)
        if p_team >= self.fav_threshold:
            w = min(w, self.fav_cap_weight)
        p = (1.0 - w) * p_prod + w * p_team
        if self.platt is not None:
            p = float(self.platt.predict_proba([[p_prod, p_team, w]])[0, 1])
        return float(np.clip(p, EPS, 1 - EPS))

    def _category_weight(self, flags: dict[str, bool]) -> float:
        if flags["both_tail"]:
            return self.weight_both_tail
        if flags["elite_involved"] and flags["cellar_involved"]:
            return self.weight_both_tail
        if flags["elite_involved"]:
            return self.weight_elite
        if flags["cellar_involved"]:
            return self.weight_cellar
        return self.weight_middle


def tail_flags(
    home_base: float,
    away_base: float,
    league_avg: float,
    *,
    elite_margin: float,
    cellar_margin: float,
) -> dict[str, bool]:
    home_hi = home_base >= league_avg + elite_margin
    home_lo = home_base <= league_avg - cellar_margin
    away_hi = away_base >= league_avg + elite_margin
    away_lo = away_base <= league_avg - cellar_margin
    elite = home_hi or away_hi
    cellar = home_lo or away_lo
    return {
        "elite_involved": elite,
        "cellar_involved": cellar,
        "both_tail": (home_hi and away_lo) or (home_lo and away_hi),
    }


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_tail_blend(
    p_prod: np.ndarray,
    p_team: np.ndarray,
    y: np.ndarray,
    home_baseline: np.ndarray,
    away_baseline: np.ndarray,
    *,
    league_avg: float,
    elite_margin: float = 0.45,
    cellar_margin: float = 0.45,
    use_platt_residual: bool = True,
) -> TailBlendModel:
    """Grid-search category weights on calibration data, optional Platt polish."""
    p_prod = np.asarray(p_prod, dtype=float)
    p_team = np.asarray(p_team, dtype=float)
    y = np.asarray(y, dtype=float)
    home_baseline = np.asarray(home_baseline, dtype=float)
    away_baseline = np.asarray(away_baseline, dtype=float)

    weight_grid = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65)
    fav_cap_grid = (0.25, 0.35, 0.45, 0.55)

    best_loss = float("inf")
    best_params = (0.25, 0.45, 0.55, 0.20, 0.45)

    for w_mid in weight_grid:
        for w_elite in weight_grid:
            for w_cellar in weight_grid:
                for w_both in weight_grid:
                    for fav_cap in fav_cap_grid:
                        if fav_cap > w_both:
                            continue
                        model = TailBlendModel(
                            league_avg_runs=league_avg,
                            league_avg_raw_runs=league_avg,
                            elite_margin=elite_margin,
                            cellar_margin=cellar_margin,
                            weight_elite=w_elite,
                            weight_cellar=w_cellar,
                            weight_both_tail=w_both,
                            weight_middle=w_mid,
                            fav_cap_weight=fav_cap,
                            platt=None,
                        )
                        p_blend = apply_tail_blend(
                            model, p_prod, p_team, home_baseline, away_baseline,
                        )
                        loss = _log_loss(y, p_blend)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = (w_mid, w_elite, w_cellar, w_both, fav_cap)

    w_mid, w_elite, w_cellar, w_both, fav_cap = best_params
    platt: LogisticRegression | None = None
    if use_platt_residual and len(y) >= 80:
        base_model = TailBlendModel(
            league_avg_runs=league_avg,
            league_avg_raw_runs=league_avg,
            elite_margin=elite_margin,
            cellar_margin=cellar_margin,
            weight_elite=w_elite,
            weight_cellar=w_cellar,
            weight_both_tail=w_both,
            weight_middle=w_mid,
            fav_cap_weight=fav_cap,
            platt=None,
        )
        p_base = apply_tail_blend(
            base_model, p_prod, p_team, home_baseline, away_baseline,
        )
        X = np.column_stack([p_prod, p_team, p_base])
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(X, y.astype(int))

    return TailBlendModel(
        league_avg_runs=league_avg,
        league_avg_raw_runs=league_avg,
        elite_margin=elite_margin,
        cellar_margin=cellar_margin,
        weight_elite=w_elite,
        weight_cellar=w_cellar,
        weight_both_tail=w_both,
        weight_middle=w_mid,
        fav_cap_weight=fav_cap,
        platt=platt,
    )


def apply_tail_blend(
    model: TailBlendModel,
    p_prod: np.ndarray,
    p_team: np.ndarray,
    home_baseline: np.ndarray,
    away_baseline: np.ndarray,
) -> np.ndarray:
    """Vectorized blended home-win probability."""
    out = np.full(len(p_prod), np.nan)
    for i in range(len(p_prod)):
        if any(np.isnan(x) for x in (p_prod[i], p_team[i], home_baseline[i], away_baseline[i])):
            continue
        flags = tail_flags(
            float(home_baseline[i]), float(away_baseline[i]), model.league_avg_runs,
            elite_margin=model.elite_margin, cellar_margin=model.cellar_margin,
        )
        w = model._category_weight(flags)
        if p_team[i] >= model.fav_threshold:
            w = min(w, model.fav_cap_weight)
        p_lin = (1.0 - w) * p_prod[i] + w * p_team[i]
        if model.platt is not None:
            p_lin = float(model.platt.predict_proba([[p_prod[i], p_team[i], p_lin]])[0, 1])
        out[i] = np.clip(p_lin, EPS, 1 - EPS)
    return out


def team_tier(
    baseline: float,
    league_avg: float,
    *,
    elite_margin: float,
    cellar_margin: float,
) -> str:
    if baseline >= league_avg + elite_margin:
        return "elite"
    if baseline <= league_avg - cellar_margin:
        return "cellar"
    return "middle"


def _side_weight(tier: str, model: TailBlendModel) -> float:
    if tier == "elite":
        return model.weight_elite
    if tier == "cellar":
        return model.weight_cellar
    return model.weight_middle


def apply_run_level_blend(
    model: TailBlendModel,
    home_runs_prod: np.ndarray,
    away_runs_prod: np.ndarray,
    home_runs_team: np.ndarray,
    away_runs_team: np.ndarray,
    home_baseline: np.ndarray,
    away_baseline: np.ndarray,
    *,
    home_baseline_raw: np.ndarray | None = None,
    away_baseline_raw: np.ndarray | None = None,
    hfa_runs: float = 0.27,
) -> np.ndarray:
    """Blend run projections per team tier, then Pythagorean win prob."""
    from src.model.evaluate import pythag_win_prob

    n = len(home_runs_prod)
    h_blend = np.full(n, np.nan)
    a_blend = np.full(n, np.nan)
    for i in range(n):
        if any(np.isnan(x) for x in (
            home_runs_prod[i], away_runs_prod[i],
            home_runs_team[i], away_runs_team[i],
            home_baseline[i], away_baseline[i],
        )):
            continue
        hb_tier = float(home_baseline_raw[i]) if home_baseline_raw is not None else float(home_baseline[i])
        ab_tier = float(away_baseline_raw[i]) if away_baseline_raw is not None else float(away_baseline[i])
        la_tier = model.league_avg_raw_runs if home_baseline_raw is not None else model.league_avg_runs
        ht = team_tier(
            hb_tier, la_tier,
            elite_margin=model.elite_margin, cellar_margin=model.cellar_margin,
        )
        at = team_tier(
            ab_tier, la_tier,
            elite_margin=model.elite_margin, cellar_margin=model.cellar_margin,
        )
        wh = _side_weight(ht, model)
        wa = _side_weight(at, model)
        if ht == "elite":
            wh = max(wh, model.min_elite_home_weight)
        if ht == "cellar":
            wh = max(wh, model.min_cellar_home_weight)
        if at == "elite":
            wa = max(wa, model.min_elite_home_weight * 0.85)
        if at == "cellar":
            wa = max(wa, model.min_cellar_home_weight * 0.85)
        h_adj = float(home_runs_prod[i]) + hfa_runs
        h_blend[i] = (1 - wh) * h_adj + wh * float(home_runs_team[i])
        a_blend[i] = (1 - wa) * float(away_runs_prod[i]) + wa * float(away_runs_team[i])

    return pythag_win_prob(h_blend, a_blend)


def fit_run_level_blend(
    home_runs_prod: np.ndarray,
    away_runs_prod: np.ndarray,
    home_runs_team: np.ndarray,
    away_runs_team: np.ndarray,
    home_baseline: np.ndarray,
    away_baseline: np.ndarray,
    y: np.ndarray,
    *,
    league_avg: float,
    league_avg_raw: float,
    home_baseline_raw: np.ndarray | None = None,
    away_baseline_raw: np.ndarray | None = None,
    elite_margin: float = 0.45,
    cellar_margin: float = 0.45,
    hfa_runs: float = 0.27,
) -> TailBlendModel:
    """Grid-search side-specific run blend weights (no Platt)."""
    weight_grid = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
    best_loss = float("inf")
    best = (0.25, 0.45, 0.35, 0.45, 0.45)

    for w_mid in weight_grid:
        for w_elite in weight_grid:
            for w_cellar in weight_grid:
                for w_both in weight_grid:
                    m = TailBlendModel(
                        league_avg_runs=league_avg,
                        league_avg_raw_runs=league_avg_raw,
                        elite_margin=elite_margin,
                        cellar_margin=cellar_margin,
                        weight_elite=w_elite,
                        weight_cellar=w_cellar,
                        weight_both_tail=w_both,
                        weight_middle=w_mid,
                        fav_cap_weight=1.0,
                        platt=None,
                    )
                    p = apply_run_level_blend(
                        m,
                        home_runs_prod, away_runs_prod,
                        home_runs_team, away_runs_team,
                        home_baseline, away_baseline,
                        home_baseline_raw=home_baseline_raw,
                        away_baseline_raw=away_baseline_raw,
                        hfa_runs=hfa_runs,
                    )
                    mask = ~np.isnan(p)
                    if mask.sum() < 50:
                        continue
                    loss = _log_loss(y[mask], p[mask])
                    if loss < best_loss:
                        best_loss = loss
                        best = (w_mid, w_elite, w_cellar, w_both, 1.0)

    w_mid, w_elite, w_cellar, w_both, fav_cap = best
    return TailBlendModel(
        league_avg_runs=league_avg,
        league_avg_raw_runs=league_avg_raw,
        elite_margin=elite_margin,
        cellar_margin=cellar_margin,
        weight_elite=w_elite,
        weight_cellar=w_cellar,
        weight_both_tail=w_both,
        weight_middle=w_mid,
        fav_cap_weight=fav_cap,
        platt=None,
        version="v2-run-level-blend",
    )


def tail_team_mask(
    home_baseline: np.ndarray,
    away_baseline: np.ndarray,
    league_avg: float,
    *,
    elite_margin: float = 0.45,
    cellar_margin: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masks for elite-involved, cellar-involved, both-tail matchups."""
    hb = np.asarray(home_baseline, dtype=float)
    ab = np.asarray(away_baseline, dtype=float)
    elite = (hb >= league_avg + elite_margin) | (ab >= league_avg + elite_margin)
    cellar = (hb <= league_avg - cellar_margin) | (ab <= league_avg - cellar_margin)
    both = (
        ((hb >= league_avg + elite_margin) & (ab <= league_avg - cellar_margin))
        | ((hb <= league_avg - cellar_margin) & (ab >= league_avg + elite_margin))
    )
    return elite, cellar, both
