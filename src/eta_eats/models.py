"""Model pipelines: preprocessing + regressor in one sklearn ``Pipeline``.

``tune`` runs a randomized search with courier-grouped folds on the training
rows only. The returned model is the refitted best estimator of that search,
so the final model is the tuned model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .features import make_preprocessor

MODELS = ("median", "ridge", "random_forest", "gbm", "mlp")
OPTIONAL = ("lightgbm",)

SEARCH_SPACE = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_iter": [200, 400, 600],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [10, 20, 50],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}


def _regressor(name: str, seed: int):
    if name == "median":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    if name == "gbm":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, random_state=seed)
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=seed)
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("the lightgbm model needs: pip install 'eta-eats[lightgbm]'") from exc
        return LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=seed, verbose=-1)
    raise ValueError(f"model must be one of {MODELS + OPTIONAL}")


def make_pipeline(name: str, seed: int = 42) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", _regressor(name, seed))])


@dataclass
class TuneResult:
    model: Pipeline
    best_params: dict
    cv_mae: float
    cv_table: pd.DataFrame


def tune(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_iter: int = 20, folds: int = 5, seed: int = 42) -> TuneResult:
    """Randomized search for the ``gbm`` pipeline with GroupKFold by courier on the training rows."""
    search = RandomizedSearchCV(
        make_pipeline("gbm", seed),
        SEARCH_SPACE,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=folds),
        random_state=seed,
        refit=True,
        n_jobs=1,
    )
    search.fit(X, y, groups=groups)
    table = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    table["mean_test_mae"] = -table.pop("mean_test_score")
    return TuneResult(search.best_estimator_, search.best_params_, float(-search.best_score_), table.sort_values("rank_test_score"))
