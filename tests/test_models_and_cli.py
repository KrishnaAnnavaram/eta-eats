"""Reference problems 2 (one pipeline), 4 (leakage), 5 (tuned model used), 6 (importance names) and 8 (seeds, test file)."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from eta_eats.cli import main
from eta_eats.clean import TARGET
from eta_eats.data import split
from eta_eats.evaluate import bootstrap_mae, distance_band, metrics
from eta_eats.features import FEATURES, model_input
from eta_eats.models import make_pipeline, tune
from eta_eats.train import compare, fit_and_score


def test_time_split_has_no_date_overlap(frame):
    parts = split(frame, "time")
    assert parts.train["order_date"].max() < parts.test["order_date"].min()


def test_courier_split_has_no_shared_courier(frame):
    parts = split(frame, "courier", seed=3)
    assert set(parts.train["courier_id"]).isdisjoint(set(parts.test["courier_id"]))


def test_splits_are_seeded(frame):
    a, b = split(frame, "random", seed=9), split(frame, "random", seed=9)
    assert a.test["id"].tolist() == b.test["id"].tolist()


def test_preprocessor_is_fitted_on_train_rows_only(frame):
    parts = split(frame, "time")
    model = make_pipeline("ridge").fit(model_input(parts.train), parts.train[TARGET])
    imputer = model.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(parts.train["courier_age"].median())
    scaler = model.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert scaler.n_samples_seen_ == len(parts.train)


def test_every_feature_reaches_the_model(frame):
    model = make_pipeline("ridge").fit(model_input(frame), frame[TARGET])
    names = model.named_steps["prep"].get_feature_names_out()
    for feature in FEATURES:
        assert any(feature in name for name in names), feature


def test_tuned_model_is_the_final_model(frame):
    parts = split(frame, "time")
    outcome = fit_and_score("gbm", parts, do_tune=True, n_iter=2, with_importance=False)
    params = outcome.model.get_params()
    for key, value in outcome.tuning["best_params"].items():
        assert params[key] == value


def test_tuning_folds_group_by_courier(frame):
    X = model_input(frame.head(600))
    result = tune(X, frame.head(600)[TARGET].to_numpy(), frame.head(600)["courier_id"].to_numpy(), n_iter=1, folds=3)
    assert len(result.cv_table) == 1 and result.cv_mae > 0


def test_models_beat_the_median_baseline(frame):
    table = compare(frame, "time", names=("median", "ridge", "gbm")).set_index("model")
    assert table.loc["gbm", "mae"] < 0.6 * table.loc["median", "mae"]
    assert table.loc["ridge", "mae"] < table.loc["median", "mae"]


def test_importance_uses_the_model_input_names(frame):
    outcome = fit_and_score("gbm", split(frame, "time"))
    assert set(outcome.importance["feature"]) == set(FEATURES)
    assert outcome.importance.iloc[0]["feature"] in {"distance_km", "traffic_x_distance"}


def test_slices_cover_traffic_and_distance(frame):
    outcome = fit_and_score("ridge", split(frame, "time"), with_importance=False)
    assert {"city", "traffic", "weather", "distance_band", "festival"} == set(outcome.slices["slice"])
    assert outcome.slices[outcome.slices["slice"] == "traffic"]["rows"].sum() == len(outcome.split.test)


def test_metrics_and_bootstrap():
    m = metrics([10, 20, 30], [12, 18, 30])
    assert m["mae"] == pytest.approx(4 / 3) and m["within_5_min_pct"] == 100
    low, high = bootstrap_mae(np.arange(100.0), np.arange(100.0) + 2)
    assert low == pytest.approx(2.0) and high == pytest.approx(2.0)
    assert distance_band(pd.Series([1.0, 4.0, 20.0, np.nan])).tolist() == ["0-3 km", "3-6 km", "15+ km", "unknown"]


def test_same_seed_gives_the_same_mlp(frame):
    parts = split(frame, "time")
    a = fit_and_score("mlp", parts, seed=1, with_importance=False).test_metrics["mae"]
    b = fit_and_score("mlp", parts, seed=1, with_importance=False).test_metrics["mae"]
    assert a == b


def test_cli_train_predict_and_model_card(data_dir, tmp_path, capsys):
    model_dir = tmp_path / "model"
    assert main(["clean", "--data", str(data_dir / "train.csv")]) == 0
    assert "pick-ups after midnight" in capsys.readouterr().out
    assert main(["train", "--data", str(data_dir / "train.csv"), "--model", "gbm", "--split", "courier", "--out", str(model_dir)]) == 0
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["split"] == "courier" and meta["features"] == list(FEATURES)
    assert "Intended use" in (model_dir / "model_card.md").read_text(encoding="utf-8")
    out = tmp_path / "sub.csv"
    assert main(["predict", "--model-dir", str(model_dir), "--input", str(data_dir / "test.csv"), "--out", str(out)]) == 0
    submission = pd.read_csv(out)
    sample = pd.read_csv(data_dir / "Sample_Submission.csv")
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample) and submission["Time_taken (min)"].between(0, 120).all()


def test_cli_compare(data_dir, tmp_path, capsys):
    assert main(["compare", "--data", str(data_dir / "train.csv"), "--models", "median", "ridge", "--out", str(tmp_path / "c.csv")]) == 0
    assert set(pd.read_csv(tmp_path / "c.csv")["model"]) == {"median", "ridge"}


def test_demo_runs_offline(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo"), "--rows", "1200", "--n-iter", "1"]) == 0
    assert "SYNTHETIC DATA" in capsys.readouterr().out


def test_lightgbm_model_is_optional(frame):
    pytest.importorskip("lightgbm")
    outcome = fit_and_score("lightgbm", split(frame, "time"), with_importance=False)
    assert outcome.test_metrics["mae"] > 0


def test_core_import_is_light():
    code = "import eta_eats.cli, sys; print([m for m in ('lightgbm', 'tensorflow', 'torch') if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
