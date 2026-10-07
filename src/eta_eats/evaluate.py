"""Evaluation: overall metrics with a bootstrap interval, slice metrics and permutation importance.

Permutation importance runs on the whole pipeline with the model input
columns, so each importance has the name of the column that was shuffled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

DISTANCE_BINS = (0, 3, 6, 10, 15, np.inf)
SLICES = ("city", "traffic", "weather", "distance_band", "festival")


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "mape_pct": float(100 * np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-9))),
        "within_5_min_pct": float(100 * np.mean(np.abs(y_pred - y_true) <= 5)),
    }


def bootstrap_mae(y_true, y_pred, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    errors = np.abs(y_pred - y_true)
    samples = errors[rng.integers(0, errors.size, size=(n_boot, errors.size))].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def distance_band(distance_km: pd.Series) -> pd.Series:
    labels = [f"{int(lo)}-{int(hi)} km" if np.isfinite(hi) else f"{int(lo)}+ km" for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])]
    return pd.cut(distance_km, bins=list(DISTANCE_BINS), labels=labels, right=False).astype("string").fillna("unknown")


def slice_metrics(frame: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    data = frame.copy()
    data["distance_band"] = distance_band(data["distance_km"])
    data["_err"] = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    data["_sq"] = (np.asarray(y_pred) - np.asarray(y_true)) ** 2
    rows = []
    for column in SLICES:
        for value, group in data.groupby(data[column].astype("string").fillna("missing"), sort=True):
            rows.append({"slice": column, "value": value, "rows": len(group), "mae": group["_err"].mean(), "rmse": np.sqrt(group["_sq"].mean())})
    return pd.DataFrame(rows)


def importance(model: Pipeline, X: pd.DataFrame, y, seed: int = 42, max_rows: int = 3000, repeats: int = 5) -> pd.DataFrame:
    if len(X) > max_rows:
        sample = X.sample(max_rows, random_state=seed)
        y = pd.Series(np.asarray(y), index=X.index).loc[sample.index].to_numpy()
        X = sample
    result = permutation_importance(model, X, y, scoring="neg_mean_absolute_error", n_repeats=repeats, random_state=seed, n_jobs=1)
    table = pd.DataFrame({"feature": list(X.columns), "mae_increase": result.importances_mean, "std": result.importances_std})
    return table.sort_values("mae_increase", ascending=False).reset_index(drop=True)
