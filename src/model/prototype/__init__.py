"""Offline prototype models for tail-aware win probability.

Does not integrate with paper trading or production inference.
See :mod:`src.model.prototype.DESIGN` (DESIGN.md) for architecture notes.
"""

from src.model.prototype.pipeline import (
    PrototypeModel,
    predict_prototype,
    train_prototype,
)
from src.model.prototype.team_baseline_model import (
    TeamBaselineRunsModel,
    predict_team_baseline_runs,
    train_team_baseline_model,
)

__all__ = [
    "PrototypeModel",
    "predict_prototype",
    "train_prototype",
    "TeamBaselineRunsModel",
    "predict_team_baseline_runs",
    "train_team_baseline_model",
]
