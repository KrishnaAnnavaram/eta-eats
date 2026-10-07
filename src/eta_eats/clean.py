"""Cleaning of the raw delivery file with explicit, checked category maps.

The raw file has trailing spaces, the string ``NaN`` for missing values, the
prefix ``conditions`` on weather values, a target stored as text (``(min) 24``),
coordinates with a wrong sign and coordinates at 0. Every category value must be
in its known set. An unknown value stops the run. It never becomes a silent NaN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "ID", "Delivery_person_ID", "Delivery_person_Age", "Delivery_person_Ratings", "Restaurant_latitude",
    "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude", "Order_Date", "Time_Orderd",
    "Time_Order_picked", "Weatherconditions", "Road_traffic_density", "Vehicle_condition", "Type_of_order",
    "Type_of_vehicle", "multiple_deliveries", "Festival", "City",
)
RAW_TARGET = "Time_taken(min)"
TARGET = "time_taken_min"

TRAFFIC_LEVELS = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
DOMAINS = {
    "weather": ("Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"),
    "traffic": tuple(TRAFFIC_LEVELS),
    "order_type": ("Snack", "Meal", "Drinks", "Buffet"),
    "vehicle": ("motorcycle", "scooter", "electric_scooter", "bicycle"),
    "festival": ("No", "Yes"),
    "city": ("Urban", "Semi-Urban", "Metropolitian"),
}
MAX_PREP_MINUTES = 120


class DataQualityError(ValueError):
    """The raw file breaks the contract (missing column, unknown category, bad target)."""


@dataclass
class CleanReport:
    rows: int
    missing: dict[str, int] = field(default_factory=dict)
    coordinate_sign_fixes: int = 0
    invalid_coordinates: int = 0
    ratings_out_of_range: int = 0
    midnight_rollovers: int = 0
    impossible_prep_times: int = 0

    def lines(self) -> list[str]:
        out = [f"rows: {self.rows}"]
        out += [f"missing {k}: {v}" for k, v in self.missing.items() if v]
        out += [
            f"coordinate sign fixes: {self.coordinate_sign_fixes}",
            f"invalid coordinates (set to missing): {self.invalid_coordinates}",
            f"ratings outside 1-5 (set to missing): {self.ratings_out_of_range}",
            f"pick-ups after midnight (rolled to the next day): {self.midnight_rollovers}",
            f"prep times above {MAX_PREP_MINUTES} min or below 0 (set to missing): {self.impossible_prep_times}",
        ]
        return out


def _strip(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.isin(["NaN", "nan", ""]))


def _category(series: pd.Series, name: str) -> pd.Series:
    values = _strip(series)
    unknown = sorted(set(values.dropna()) - set(DOMAINS[name]))
    if unknown:
        raise DataQualityError(f"unknown {name} value(s): {unknown}")
    return values


def parse_target(series: pd.Series) -> pd.Series:
    """``(min) 24`` -> 24.0 with an explicit regular expression."""
    def one(value):
        match = re.search(r"(\d+(?:\.\d+)?)", str(value))
        return float(match.group(1)) if match else np.nan

    return series.map(one).astype(float)


def _times(dates: pd.Series, times: pd.Series) -> pd.Series:
    text = _strip(times)
    stamp = dates.dt.strftime("%Y-%m-%d") + " " + text
    return pd.to_datetime(stamp, format="mixed", errors="coerce")


def courier_city(courier_id: pd.Series) -> pd.Series:
    """The city code at the start of the courier ID, for example ``INDORES13DEL02`` -> ``INDO``."""
    return courier_id.str.extract(r"^([A-Z]+?)RES", expand=False).fillna("UNKNOWN")


def clean(raw: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, CleanReport]:
    missing_cols = [c for c in RAW_COLUMNS if c not in raw.columns]
    if require_target and RAW_TARGET not in raw.columns:
        missing_cols.append(RAW_TARGET)
    if missing_cols:
        raise DataQualityError(f"missing columns: {missing_cols}")
    report = CleanReport(rows=len(raw))
    out = pd.DataFrame(index=raw.index)
    out["id"] = _strip(raw["ID"])
    out["courier_id"] = _strip(raw["Delivery_person_ID"])
    out["courier_city"] = courier_city(out["courier_id"].fillna(""))
    out["courier_age"] = pd.to_numeric(_strip(raw["Delivery_person_Age"]), errors="coerce")
    rating = pd.to_numeric(_strip(raw["Delivery_person_Ratings"]), errors="coerce")
    bad_rating = (rating < 1) | (rating > 5)
    report.ratings_out_of_range = int(bad_rating.sum())
    out["courier_rating"] = rating.mask(bad_rating)

    for raw_col, col in (("Restaurant_latitude", "rest_lat"), ("Restaurant_longitude", "rest_lon"),
                         ("Delivery_location_latitude", "drop_lat"), ("Delivery_location_longitude", "drop_lon")):
        values = pd.to_numeric(raw[raw_col], errors="coerce")
        report.coordinate_sign_fixes += int((values < 0).sum())
        values = values.abs()
        invalid = values < 1.0
        report.invalid_coordinates += int(invalid.sum())
        out[col] = values.mask(invalid)

    out["order_date"] = pd.to_datetime(_strip(raw["Order_Date"]), format="%d-%m-%Y", errors="coerce")
    if out["order_date"].isna().any():
        raise DataQualityError(f"{int(out['order_date'].isna().sum())} row(s) with an unreadable Order_Date")
    out["order_time"] = _times(out["order_date"], raw["Time_Orderd"])
    picked = _times(out["order_date"], raw["Time_Order_picked"])
    rollover = picked.notna() & out["order_time"].notna() & (picked < out["order_time"])
    report.midnight_rollovers = int(rollover.sum())
    out["pickup_time"] = picked.where(~rollover, picked + pd.Timedelta(days=1))

    out["weather"] = _category(raw["Weatherconditions"].astype("string").str.replace("conditions", "", regex=False), "weather")
    out["traffic"] = _category(raw["Road_traffic_density"], "traffic")
    out["traffic_level"] = out["traffic"].map(TRAFFIC_LEVELS).astype(float)
    if out["traffic_level"].isna().sum() != out["traffic"].isna().sum():
        raise DataQualityError("traffic map created missing values")
    out["vehicle_condition"] = pd.to_numeric(_strip(raw["Vehicle_condition"]), errors="coerce")
    out["order_type"] = _category(raw["Type_of_order"], "order_type")
    out["vehicle"] = _category(raw["Type_of_vehicle"], "vehicle")
    out["multiple_deliveries"] = pd.to_numeric(_strip(raw["multiple_deliveries"]), errors="coerce")
    out["festival"] = _category(raw["Festival"], "festival")
    out["city"] = _category(raw["City"], "city")

    if RAW_TARGET in raw.columns:
        out[TARGET] = parse_target(raw[RAW_TARGET])
        if require_target and out[TARGET].isna().any():
            raise DataQualityError(f"{int(out[TARGET].isna().sum())} row(s) with an unreadable target")
    report.missing = {c: int(out[c].isna().sum()) for c in out.columns if c not in ("id",)}
    return out.reset_index(drop=True), report
