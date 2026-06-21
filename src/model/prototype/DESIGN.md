# Prototype model v1 — tail-aware win probability

**Status:** research / offline only. Does **not** feed paper trading, bet log,
dashboard, or Lambda inference.

## Problem

Production uses Ridge → point run estimates → Pythagorean `p_home` → EV bets.
That pipeline compresses team tails (Rockies, Dodgers) and ignores run-scoring
variance. Paper trading stays on the production path until this prototype proves
value on held-out data.

## Production path (unchanged)

```
build_features → train_runs_model → predict_runs → pythag_win_prob
    → annotate_bets → bet_log → dashboard
```

## Prototype path (new, parallel)

```
build_features → train_runs_model (same μ) → predict_runs
    → NB Monte Carlo P(home wins) → tail isotonic calibration
    → compare vs baseline (offline report only)

build_features → team RS 30d (park-adj, shrunk) + Ridge residuals (SP/lineup)
    → Pythag → compare vs production (offline report only)
```

### Team RS baseline (v2)

Anchor (per side, home/away split):

    w = min(1, n_games_in_30d_window / 30)
    μ_anchor = w · team_RS_30d_park_adj + (1 − w) · league_avg

Ridge fits residuals only:

    μ = μ_anchor + Ridge(lineup_delta, SP, BP, ...)

Park adjustment divides raw runs by venue `index_runs` (mean R/L).
Bet-eligible when **both** teams have ≥30 season games played.

### Tail blend (v3)

Run-level blend per team tier (elite / cellar / middle), using **raw** 30d RS
for tier detection (park adj hides Coors cellar). Weights fit on cal slice;
home-side floors (`elite` ≥50%, `cellar` ≥55%) keep tail teams from reverting
to production.

    μ_home = (1−w_home)·μ_prod + w_home·μ_team     (tier from raw RS)
    p_home = Pythag(μ_home, μ_away)

Optional v2 branches (not all implemented yet):

| Branch | Idea | Tail benefit |
|--------|------|--------------|
| **NB + MC** (v1) | μ from Ridge; dispersion α from residual variance | Coors / high-scoring games get wider distributions |
| **Tail isotonic** (v1) | Separate calibration for dogs / mids / favorites | Fixes LAD -220 and COL dog miscalibration |
| **Win classifier** (v2) | Logistic on game-level feature diffs → `home_win` | Learns asymmetric favorite/dog mapping |
| **Team intercepts** (v2) | Shrunk offense/defense residuals per team | Less Ridge compression on COL/LAD |
| **Market blend** (v2) | `p = w·p_model + (1-w)·p_market` when \|p-0.5\| large | Reduces false edge vs efficient markets |

## Evaluation protocol

Run on the same split as production (train 2023+2024, test 2025):

1. **Global:** log loss, Brier, accuracy
2. **Calibration:** 10 quantile buckets + favorite (p>0.60) / dog (p<0.40) slices
3. **Team tails:** mean `(p_home - home_win)` when Rockies or Dodgers involved
4. **Runs:** MAE unchanged (same μ model)
5. **Betting sim (optional):** if historical odds joined, ROI vs baseline at same EV threshold

## Usage

```bash
# Requires data/features/training_{2023,2024,2025}.parquet
make prototype-compare

# Or directly:
.venv/bin/python -m src.model.prototype.compare \
  --train-years 2023 2024 --test-year 2025 \
  --out data/research/prototype_compare.md
```

Output: `data/research/prototype_compare.md` (never touches `bet_log.parquet`).

## Promotion criteria (future)

Before any prototype replaces production:

- Lower log loss on 2025 holdout
- Better favorite/dog calibration gaps (not just overall)
- Non-negative or improved ROI in backtest at same stake rules
- Explicit sign-off + separate model artifact path (e.g. `runs_model_prototype.pkl`)
