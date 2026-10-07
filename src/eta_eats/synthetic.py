"""Synthetic deliveries in the raw Kaggle format, with the same quirks. All values are generated.

The delivery time follows a known formula of distance, traffic (also as a
traffic x distance interaction), meal-time peaks, weather, preparation time,
multiple deliveries, festival days, vehicle condition and the courier. The file keeps the raw quirks: trailing spaces, ``NaN`` strings, the
``conditions`` prefix, ``(min) N`` targets, wrong coordinate signs, zero
coordinates and pick-ups after midnight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CITIES = {  # code: (lat, lon, city type)
    "INDO": (22.72, 75.86, "Urban"), "BANG": (12.97, 77.59, "Metropolitian"), "MUM": (19.08, 72.88, "Metropolitian"),
    "CHEN": (13.08, 80.27, "Metropolitian"), "HYD": (17.39, 78.49, "Metropolitian"), "PUNE": (18.52, 73.86, "Urban"),
    "JAP": (26.91, 75.79, "Urban"), "SUR": (21.17, 72.83, "Urban"), "KOC": (9.93, 76.27, "Semi-Urban"),
}
TRAFFIC_MIN = {"Low": 0.0, "Medium": 4.0, "High": 7.0, "Jam": 11.0}
WEATHER_MIN = {"Sunny": 0.0, "Cloudy": 2.0, "Windy": 2.0, "Fog": 5.0, "Stormy": 5.0, "Sandstorms": 4.0}
TRAFFIC_P = [0.34, 0.24, 0.10, 0.32]


def _maybe_nan(rng, values: list[str], share: float) -> list[str]:
    return [("NaN " if rng.random() < share else v) for v in values]


def generate(n: int = 6000, seed: int = 42, with_target: bool = True, id_offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = list(CITIES)
    city_code = rng.choice(codes, size=n)
    couriers = {c: [f"{c}RES{r:02d}DEL{d:02d}" for r in range(1, 21) for d in (1, 2, 3)] for c in codes}
    courier = np.array([rng.choice(couriers[c]) for c in city_code])
    courier_skill = {cid: rng.normal(0, 1.5) for ids in couriers.values() for cid in ids}

    base_lat = np.array([CITIES[c][0] for c in city_code]) + rng.normal(0, 0.03, n)
    base_lon = np.array([CITIES[c][1] for c in city_code]) + rng.normal(0, 0.03, n)
    step = rng.uniform(0.01, 0.13, n)
    drop_lat = base_lat + step * rng.choice([-1, 1], n)
    drop_lon = base_lon + step * rng.choice([-1, 1], n)
    distance = 111.0 * np.hypot(drop_lat - base_lat, (drop_lon - base_lon) * np.cos(np.radians(base_lat)))

    dates = pd.Timestamp("2022-02-11") + pd.to_timedelta(rng.integers(0, 55, n), unit="D")
    order_min = rng.choice(np.r_[np.arange(8 * 60, 23 * 60 + 55, 5), np.arange(0, 60, 5)], size=n)
    prep = rng.choice([5, 10, 15], size=n)
    pick_min = (order_min + prep) % (24 * 60)

    traffic = rng.choice(list(TRAFFIC_MIN), size=n, p=TRAFFIC_P)
    weather = rng.choice(list(WEATHER_MIN), size=n)
    multiple = rng.choice([0, 1, 2, 3], size=n, p=[0.31, 0.62, 0.05, 0.02])
    festival = np.where(rng.random(n) < 0.02, "Yes", "No")
    vehicle_condition = rng.choice([0, 1, 2, 3], size=n, p=[0.33, 0.33, 0.33, 0.01])
    vehicle = rng.choice(["motorcycle", "scooter", "electric_scooter", "bicycle"], size=n, p=[0.58, 0.335, 0.083, 0.002])
    age = rng.integers(20, 40, n)
    rating = np.round(np.clip(rng.normal(4.6, 0.3, n), 2.5, 5.0), 1)

    minutes = (
        10 + 1.6 * distance + np.vectorize(TRAFFIC_MIN.get)(traffic) + np.vectorize(WEATHER_MIN.get)(weather)
        + 0.3 * prep + 3.0 * multiple + 8.0 * (festival == "Yes") - 1.5 * vehicle_condition
        + 0.12 * (age - 30) - 4.0 * (rating - 4.6) + np.array([courier_skill[c] for c in courier]) + rng.normal(0, 3.0, n)
        + 0.35 * distance * np.vectorize(list(TRAFFIC_MIN).index)(traffic)
        + 4.0 * (((order_min >= 12 * 60) & (order_min < 14 * 60)) | ((order_min >= 19 * 60) & (order_min < 22 * 60)))
    )
    minutes = np.clip(np.round(minutes), 10, 54).astype(int)

    rest_lat, rest_lon = base_lat.copy(), base_lon.copy()
    flip = rng.random(n) < 0.01
    rest_lat[flip] *= -1
    zero = rng.random(n) < 0.03
    rest_lat[zero], rest_lon[zero] = 0.0, 0.0

    def clock(m):
        return f"{int(m) // 60:02d}:{int(m) % 60:02d}:00"

    frame = pd.DataFrame(
        {
            "ID": [f"0x{id_offset + i:04x} " for i in range(n)],
            "Delivery_person_ID": [f"{c} " for c in courier],
            "Delivery_person_Age": _maybe_nan(rng, [str(a) for a in age], 0.04),
            "Delivery_person_Ratings": _maybe_nan(rng, [str(r) for r in rating], 0.04),
            "Restaurant_latitude": np.round(rest_lat, 6),
            "Restaurant_longitude": np.round(rest_lon, 6),
            "Delivery_location_latitude": np.round(drop_lat, 6),
            "Delivery_location_longitude": np.round(drop_lon, 6),
            "Order_Date": dates.strftime("%d-%m-%Y"),
            "Time_Orderd": _maybe_nan(rng, [clock(m) for m in order_min], 0.04),
            "Time_Order_picked": [clock(m) for m in pick_min],
            "Weatherconditions": [f"conditions {w}" if rng.random() > 0.015 else "conditions NaN" for w in weather],
            "Road_traffic_density": _maybe_nan(rng, [f"{t} " for t in traffic], 0.013),
            "Vehicle_condition": vehicle_condition.astype(str),
            "Type_of_order": [f"{t} " for t in rng.choice(["Snack", "Meal", "Drinks", "Buffet"], size=n)],
            "Type_of_vehicle": [f"{v} " for v in vehicle],
            "multiple_deliveries": _maybe_nan(rng, [str(m) for m in multiple], 0.02),
            "Festival": _maybe_nan(rng, [f"{f} " for f in festival], 0.005),
            "City": _maybe_nan(rng, [f"{CITIES[c][2]} " for c in city_code], 0.026),
        }
    )
    if with_target:
        frame["Time_taken(min)"] = [f"(min) {m}" for m in minutes]
    return frame


def write_synthetic(out_dir, n_train: int = 6000, n_test: int = 1500, seed: int = 42) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"train": out / "train.csv", "test": out / "test.csv", "sample": out / "Sample_Submission.csv"}
    generate(n_train, seed).to_csv(paths["train"], index=False)
    test = generate(n_test, seed + 1, with_target=False, id_offset=n_train)
    test.to_csv(paths["test"], index=False)
    pd.DataFrame({"ID": test["ID"], "Time_taken (min)": 0.0}).to_csv(paths["sample"], index=False)
    return paths
