"""Training, model comparison, saving, the model card and test-file prediction."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .clean import TARGET
from .data import SplitResult, load, split
from .evaluate import bootstrap_mae, importance, metrics, slice_metrics
from .features import FEATURES, model_input
from .models import MODELS, make_pipeline, tune

SUBMISSION_TARGET = "Time_taken (min)"


@dataclass
class TrainOutcome:
    name: str
    model: object
    split: SplitResult
    test_metrics: dict
    mae_ci: tuple[float, float]
    slices: pd.DataFrame
    importance: pd.DataFrame
    tuning: dict | None


def fit_and_score(name: str, parts: SplitResult, seed: int = 42, do_tune: bool = False, n_iter: int = 20,
                  with_importance: bool = True) -> TrainOutcome:
    X_train, y_train = model_input(parts.train), parts.train[TARGET].to_numpy()
    X_test, y_test = model_input(parts.test), parts.test[TARGET].to_numpy()
    tuning = None
    if do_tune:
        if name != "gbm":
            raise ValueError("--tune is available for the gbm model")
        result = tune(X_train, y_train, parts.train["courier_id"].fillna("missing").to_numpy(), n_iter=n_iter, seed=seed)
        model = result.model
        tuning = {"best_params": result.best_params, "cv_mae": result.cv_mae, "n_iter": n_iter}
    else:
        model = make_pipeline(name, seed).fit(X_train, y_train)
    pred = model.predict(X_test)
    imp = importance(model, X_test, y_test, seed=seed) if with_importance else pd.DataFrame()
    return TrainOutcome(name, model, parts, metrics(y_test, pred), bootstrap_mae(y_test, pred, seed=seed),
                        slice_metrics(parts.test, y_test, pred), imp, tuning)


def compare(frame: pd.DataFrame, split_kind: str = "time", seed: int = 42, names=MODELS) -> pd.DataFrame:
    parts = split(frame, split_kind, seed=seed)
    rows = []
    for name in names:
        outcome = fit_and_score(name, parts, seed=seed, with_importance=False)
        low, high = outcome.mae_ci
        rows.append({"model": name, **outcome.test_metrics, "mae_ci_low": low, "mae_ci_high": high})
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def model_card(outcome: TrainOutcome, data_path: str, report_lines: list[str]) -> str:
    m = outcome.test_metrics
    low, high = outcome.mae_ci
    top = outcome.importance.head(5)
    lines = [
        f"# Model card: eta-eats `{outcome.name}`",
        "",
        "## Intended use",
        "",
        "Estimate the delivery time of a food order in minutes, for planning and for customer messages.",
        "Do not use the estimate to judge or to pay a courier.",
        "",
        "## Data",
        "",
        f"- File: `{data_path}`",
        f"- Split: `{outcome.split.kind}` ({outcome.split.description})",
        f"- Train rows: {len(outcome.split.train)}. Test rows: {len(outcome.split.test)}",
        *[f"- {line}" for line in report_lines],
        "",
        "## Test metrics",
        "",
        f"- MAE {m['mae']:.2f} min (95 % bootstrap interval {low:.2f} to {high:.2f})",
        f"- RMSE {m['rmse']:.2f} min, R2 {m['r2']:.3f}, within 5 min {m['within_5_min_pct']:.1f} %",
        "",
        "## Most important inputs (permutation importance, MAE increase in minutes)",
        "",
        *[f"- `{r.feature}`: {r.mae_increase:.2f}" for r in top.itertuples()],
        "",
        "## Limits",
        "",
        "- The model learns from one period and one set of cities. Check the slice table before you use it elsewhere.",
        "- Courier IDs are not inputs. The courier city is an input.",
        "",
    ]
    if outcome.tuning:
        lines += ["## Tuning", "", f"- Best parameters: `{outcome.tuning['best_params']}`", f"- Cross-validated MAE: {outcome.tuning['cv_mae']:.2f}", ""]
    return "\n".join(lines)


def save(outcome: TrainOutcome, out_dir, data_path: str, report_lines: list[str], seed: int) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(outcome.model, out / "model.joblib")
    meta = {
        "model": outcome.name,
        "features": list(FEATURES),
        "split": outcome.split.kind,
        "split_description": outcome.split.description,
        "seed": seed,
        "test_metrics": outcome.test_metrics,
        "mae_ci95": list(outcome.mae_ci),
        "tuning": outcome.tuning,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    outcome.slices.to_csv(out / "slices.csv", index=False)
    outcome.importance.to_csv(out / "importance.csv", index=False)
    (out / "model_card.md").write_text(model_card(outcome, data_path, report_lines), encoding="utf-8")
    return out


def load_model(model_dir):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta["features"] != list(FEATURES):
        raise ValueError("the saved model uses other features than this version of eta-eats: train it again")
    return joblib.load(model_dir / "model.joblib"), meta


def predict_file(model_dir, input_path, out_path) -> pd.DataFrame:
    """Score a file without a target (for example the Kaggle ``test.csv``) and write a submission."""
    model, _ = load_model(model_dir)
    frame, _ = load(input_path, require_target=False)
    pred = np.clip(model.predict(model_input(frame)), 0, None)
    submission = pd.DataFrame({"ID": frame["id"], SUBMISSION_TARGET: np.round(pred, 4)})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    return submission
