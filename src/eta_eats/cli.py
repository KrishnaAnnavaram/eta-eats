"""Command line interface: ``eta-eats <command>``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from . import __version__
from .data import SPLITS, load, split
from .models import MODELS, OPTIONAL
from .synthetic import write_synthetic
from .train import compare, fit_and_score, predict_file, save


def _env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _seed(args) -> int:
    if getattr(args, "seed", None) is not None:
        return args.seed
    raw = _env("ETA_EATS_SEED", "42")
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"ETA_EATS_SEED must be an integer, got {raw!r}") from exc


def _data_path(args) -> Path:
    return Path(args.data or Path(_env("ETA_EATS_DATA_DIR", "data")) / "train.csv")


def _print(frame: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.to_string(index=False))


def cmd_synth(args) -> int:
    for name, path in write_synthetic(args.out, n_train=args.rows, n_test=args.test_rows, seed=args.seed).items():
        print(f"{name}: {path}")
    print("SYNTHETIC DATA: generated deliveries, not real orders.")
    return 0


def cmd_clean(args) -> int:
    frame, report = load(_data_path(args), require_target=not args.no_target)
    for line in report.lines():
        print(line)
    print(f"traffic levels: {frame['traffic'].value_counts(dropna=False).to_dict()}")
    return 0


def cmd_train(args) -> int:
    seed = _seed(args)
    path = _data_path(args)
    frame, report = load(path)
    parts = split(frame, args.split, seed=seed)
    outcome = fit_and_score(args.model, parts, seed=seed, do_tune=args.tune, n_iter=args.n_iter)
    out = save(outcome, args.out or _env("ETA_EATS_MODEL_DIR", "models"), str(path), report.lines(), seed)
    low, high = outcome.mae_ci
    print(f"split: {parts.kind} ({parts.description})")
    print(json.dumps({k: round(v, 4) for k, v in outcome.test_metrics.items()}))
    print(f"MAE 95 % bootstrap interval: {low:.3f} to {high:.3f}")
    if outcome.tuning:
        print(f"tuned: {outcome.tuning['best_params']} (cross-validated MAE {outcome.tuning['cv_mae']:.3f})")
    print(f"saved model, metadata, slices, importance and model card in {out}")
    return 0


def cmd_compare(args) -> int:
    frame, _ = load(_data_path(args))
    table = compare(frame, args.split, seed=_seed(args), names=args.models)
    _print(table.round(3))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    return 0


def cmd_predict(args) -> int:
    out = args.out or str(Path(_env("ETA_EATS_OUTPUT_DIR", "reports")) / "submission.csv")
    submission = predict_file(args.model_dir or _env("ETA_EATS_MODEL_DIR", "models"), args.input, out)
    print(f"wrote {len(submission)} predictions to {out}")
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out)
    paths = write_synthetic(out / "data", n_train=args.rows, n_test=1500, seed=42)
    print("SYNTHETIC DATA: generated deliveries, not real orders.\n")
    frame, report = load(paths["train"])
    for line in report.lines():
        print(line)
    for kind in ("time", "courier"):
        print(f"\nModel comparison, {kind} split:")
        _print(compare(frame, kind, seed=42).round(3))
    parts = split(frame, "time", seed=42)
    outcome = fit_and_score("gbm", parts, seed=42, do_tune=True, n_iter=args.n_iter)
    save(outcome, out / "model", str(paths["train"]), report.lines(), 42)
    print(f"\nTuned gbm on the time split: MAE {outcome.test_metrics['mae']:.3f}, "
          f"cross-validated MAE {outcome.tuning['cv_mae']:.3f}, best {outcome.tuning['best_params']}")
    print("\nTop permutation importance (MAE increase, minutes):")
    _print(outcome.importance.head(6).round(3))
    print("\nMAE by traffic level:")
    _print(outcome.slices[outcome.slices["slice"] == "traffic"].round(3))
    submission = predict_file(out / "model", paths["test"], out / "submission.csv")
    print(f"\nScored {len(submission)} rows of the test file: {out / 'submission.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eta-eats", description="Food delivery time prediction")
    parser.add_argument("--version", action="version", version=f"eta-eats {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write synthetic train.csv, test.csv and Sample_Submission.csv")
    p.add_argument("--out", default="data")
    p.add_argument("--rows", type=int, default=6000)
    p.add_argument("--test-rows", dest="test_rows", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("clean", help="data quality report of a raw file")
    p.add_argument("--data")
    p.add_argument("--no-target", dest="no_target", action="store_true")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("train", help="train one model, evaluate it and save it with a model card")
    p.add_argument("--data")
    p.add_argument("--model", choices=MODELS + OPTIONAL, default="gbm")
    p.add_argument("--split", choices=SPLITS, default="time")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-iter", dest="n_iter", type=int, default=20)
    p.add_argument("--seed", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("compare", help="all models on one split")
    p.add_argument("--data")
    p.add_argument("--split", choices=SPLITS, default="time")
    p.add_argument("--models", nargs="+", choices=MODELS + OPTIONAL, default=list(MODELS))
    p.add_argument("--seed", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("predict", help="score a file without a target and write a submission CSV")
    p.add_argument("--model-dir", dest="model_dir")
    p.add_argument("--input", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("demo", help="synthetic data, comparison, tuned model, prediction, offline")
    p.add_argument("--out", default="reports/demo")
    p.add_argument("--rows", type=int, default=6000)
    p.add_argument("--n-iter", dest="n_iter", type=int, default=8)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
