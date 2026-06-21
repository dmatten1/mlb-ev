"""Smoke tests for offline prototype (no production path touched)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.prototype.nb_win_prob import estimate_dispersion, simulate_p_home
from src.model.prototype.pipeline import predict_prototype, train_prototype
from src.model.prototype.tail_calibrator import fit_tail_calibration, tail_calibration_table
from src.model.runs_model import BULLPEN_FEATURE_COLS


def _synthetic_games(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h_off = 0.32 + rng.normal(0, 0.02)
        a_off = 0.31 + rng.normal(0, 0.02)
        h_sp = 3.8 + rng.normal(0, 0.2)
        a_sp = 3.9 + rng.normal(0, 0.2)
        home_runs = max(0, int(4.2 + 8 * (h_off - 0.30) - 0.3 * (a_sp - 3.8) + rng.normal(0, 1.5)))
        away_runs = max(0, int(4.0 + 8 * (a_off - 0.30) - 0.3 * (h_sp - 3.8) + rng.normal(0, 1.5)))
        row = {
            "game_id": i,
            "game_date": f"2024-06-{(i % 28) + 1:02d}",
            "season_year": 2024,
            "home_name": "Team A",
            "away_name": "Team B",
            "home_score": home_runs,
            "away_score": away_runs,
            "home_win": int(home_runs > away_runs),
        }
        for feat in BULLPEN_FEATURE_COLS:
            if feat == "is_home":
                continue
            if feat.startswith("off_"):
                row[f"home_{feat}"] = h_off + rng.normal(0, 0.005)
                row[f"away_{feat}"] = a_off + rng.normal(0, 0.005)
            elif feat.startswith("opp_sp_"):
                sp_feat = f"sp_{feat[len('opp_sp_'):]}"
                row[f"away_{sp_feat}"] = a_sp + rng.normal(0, 0.05)
                row[f"home_{sp_feat}"] = h_sp + rng.normal(0, 0.05)
            elif feat.startswith("opp_bp_"):
                bp_feat = f"bp_{feat[len('opp_bp_'):]}"
                row[f"away_{bp_feat}"] = 0.32
                row[f"home_{bp_feat}"] = 0.32
        rows.append(row)
    return pd.DataFrame(rows)


def test_dispersion_and_simulation():
    mu_h = np.array([4.5, 3.8])
    mu_a = np.array([3.9, 4.6])
    alpha = 4.0
    p = simulate_p_home(mu_h, mu_a, alpha=alpha, n_sim=5000, seed=1)
    assert p.shape == (2,)
    assert 0.0 < p[0] < 1.0
    assert 0.0 < p[1] < 1.0
    assert p[0] > p[1]


def test_train_and_predict_prototype():
    games = _synthetic_games(160)
    cal = games.iloc[120:].copy()
    train = games.iloc[:120].copy()
    model = train_prototype(train, cal, n_sim=1000)
    pred = predict_prototype(games.iloc[120:].copy(), model)
    for col in ("p_home_pythag", "p_home_nb", "p_home_proto_cal"):
        assert col in pred.columns
        assert pred[col].between(0, 1).all()


def test_tail_calibration_table():
    p = np.linspace(0.2, 0.8, 200)
    y = (p + np.random.default_rng(0).normal(0, 0.1, 200) > 0.5).astype(float)
    cal = fit_tail_calibration(p, y, min_region_n=20)
    out = cal.transform(p)
    tbl = tail_calibration_table(out, y)
    assert len(tbl) == 3
