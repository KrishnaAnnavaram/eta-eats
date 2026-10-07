"""Row-wise features and the one model pipeline.

``add_features`` calculates values from one row only (distance, preparation
time, hour, weekday, the traffic x distance interaction). It has no fitted
state, so it cannot leak. ``make_preprocessor`` builds the ColumnTransformer
(imputation, scaling, one-hot encoding). The model pipeline fits it on the
training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .clean import MAX_PREP_MINUTES

EARTH_RADIUS_KM = 6371.0088
NUMERIC = (
    "courier_age", "courier_rating", "distance_km", "prep_min", "order_hour", "weekday", "vehicle_condition",
    "multiple_deliveries", "traffic_level", "traffic_x_distance",
)
CATEGORICAL = ("weather", "traffic", "order_type", "vehicle", "festival", "city", "courier_city")
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = ("id", "courier_id")  # identifiers are never features


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return the frame with the row-wise features, and the count of impossible prep times."""
    out = frame.copy()
    out["distance_km"] = haversine_km(out["rest_lat"], out["rest_lon"], out["drop_lat"], out["drop_lon"])
    prep = (out["pickup_time"] - out["order_time"]).dt.total_seconds() / 60.0
    impossible = (prep < 0) | (prep > MAX_PREP_MINUTES)
    out["prep_min"] = prep.mask(impossible)
    reference = out["order_time"].fillna(out["pickup_time"])
    out["order_hour"] = reference.dt.hour + reference.dt.minute / 60.0
    out["weekday"] = out["order_date"].dt.weekday.astype(float)
    out["traffic_x_distance"] = out["traffic_level"] * out["distance_km"]
    return out, int(impossible.sum())


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", numeric, list(NUMERIC)), ("cat", categorical, list(CATEGORICAL))])


def model_input(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns that the pipeline gets. Categories are plain objects for the imputer."""
    data = frame[list(FEATURES)].copy()
    for col in CATEGORICAL:
        data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
    return data
