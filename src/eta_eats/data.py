"""Loading and splitting: time-based, grouped by courier, or random. All splits are seeded."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .clean import CleanReport, clean
from .features import add_features

SPLITS = ("time", "courier", "random")


def prepare(raw: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    frame, report = clean(raw, require_target=require_target)
    frame, impossible = add_features(frame)
    report.impossible_prep_times = impossible
    return frame, report


def load(path, require_target: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    return prepare(pd.read_csv(path, dtype=str, keep_default_na=False), require_target=require_target)


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    kind: str
    description: str


def split(frame: pd.DataFrame, kind: str = "time", test_size: float = 0.2, seed: int = 42) -> SplitResult:
    """``time``: the latest order dates are the test rows. ``courier``: no courier is in both parts."""
    if kind not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if kind == "time":
        dates = np.sort(frame["order_date"].unique())
        cut = dates[int(np.floor(len(dates) * (1 - test_size)))]
        train, test = frame[frame["order_date"] < cut], frame[frame["order_date"] >= cut]
        text = f"train orders before {pd.Timestamp(cut).date()}, test orders from {pd.Timestamp(cut).date()}"
    elif kind == "courier":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(frame, groups=frame["courier_id"].fillna("missing")))
        train, test = frame.iloc[tr], frame.iloc[te]
        text = f"{train['courier_id'].nunique()} couriers in train, {test['courier_id'].nunique()} other couriers in test"
    else:
        train, test = train_test_split(frame, test_size=test_size, random_state=seed)
        text = "random rows, seeded"
    if train.empty or test.empty:
        raise ValueError("the split gave an empty part")
    return SplitResult(train.reset_index(drop=True), test.reset_index(drop=True), kind, text)
