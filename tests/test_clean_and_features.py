"""Reference problems 1 (traffic map), 3 (midnight prep time), 7 (courier ID) and 9 (target parsing)."""

import numpy as np
import pandas as pd
import pytest

from eta_eats.clean import TRAFFIC_LEVELS, DataQualityError, clean, courier_city, parse_target
from eta_eats.features import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC, add_features, haversine_km


def _row(**changes):
    base = {
        "ID": "0x1 ", "Delivery_person_ID": "INDORES13DEL02 ", "Delivery_person_Age": "37", "Delivery_person_Ratings": "4.9",
        "Restaurant_latitude": "22.745049", "Restaurant_longitude": "75.892471", "Delivery_location_latitude": "22.765049",
        "Delivery_location_longitude": "75.912471", "Order_Date": "19-03-2022", "Time_Orderd": "11:30:00",
        "Time_Order_picked": "11:45:00", "Weatherconditions": "conditions Sunny", "Road_traffic_density": "High ",
        "Vehicle_condition": "2", "Type_of_order": "Snack ", "Type_of_vehicle": "motorcycle ", "multiple_deliveries": "0",
        "Festival": "No ", "City": "Urban ", "Time_taken(min)": "(min) 24",
    }
    base.update(changes)
    return pd.DataFrame([base])


@pytest.mark.parametrize("raw_value, level", [("Low ", 0), ("Medium ", 1), ("High ", 2), ("Jam ", 3)])
def test_every_traffic_level_is_mapped(raw_value, level):
    frame, _ = clean(_row(Road_traffic_density=raw_value))
    assert frame.loc[0, "traffic_level"] == level
    assert list(TRAFFIC_LEVELS) == ["Low", "Medium", "High", "Jam"]


def test_unknown_category_stops_the_run():
    with pytest.raises(DataQualityError, match="unknown traffic"):
        clean(_row(Road_traffic_density="jammed"))
    with pytest.raises(DataQualityError, match="unknown weather"):
        clean(_row(Weatherconditions="conditions Snow"))


def test_nan_strings_become_missing_not_errors():
    frame, report = clean(_row(Road_traffic_density="NaN ", Weatherconditions="conditions NaN", Delivery_person_Age="NaN "))
    assert pd.isna(frame.loc[0, "traffic"]) and pd.isna(frame.loc[0, "traffic_level"])
    assert pd.isna(frame.loc[0, "weather"]) and pd.isna(frame.loc[0, "courier_age"])
    assert report.missing["traffic"] == 1


def test_traffic_counts_survive_cleaning(raw, frame):
    raw_counts = raw["Road_traffic_density"].str.strip().value_counts()
    for level in TRAFFIC_LEVELS:
        assert (frame["traffic"] == level).sum() == raw_counts[level]


def test_target_parsing():
    assert parse_target(pd.Series(["(min) 24", "(min) 9", "33"])).tolist() == [24.0, 9.0, 33.0]
    with pytest.raises(DataQualityError, match="target"):
        clean(_row(**{"Time_taken(min)": "(min) ?"}))


def test_pickup_after_midnight_rolls_to_the_next_day():
    frame, report = clean(_row(Time_Orderd="23:50:00", Time_Order_picked="00:05:00"))
    frame, impossible = add_features(frame)
    assert report.midnight_rollovers == 1
    assert frame.loc[0, "prep_min"] == pytest.approx(15.0)
    assert impossible == 0


def test_dates_come_from_the_order_date_not_today():
    frame, _ = clean(_row())
    assert frame.loc[0, "order_time"] == pd.Timestamp("2022-03-19 11:30:00")
    frame, _ = add_features(frame)
    assert frame.loc[0, "weekday"] == 5 and frame.loc[0, "order_hour"] == pytest.approx(11.5)


def test_coordinates_sign_fix_and_zero_coordinates():
    frame, report = clean(_row(Restaurant_latitude="-22.745049"))
    assert frame.loc[0, "rest_lat"] == pytest.approx(22.745049) and report.coordinate_sign_fixes == 1
    frame, report = clean(_row(Restaurant_latitude="0.0", Restaurant_longitude="0.0"))
    frame, _ = add_features(frame)
    assert pd.isna(frame.loc[0, "distance_km"]) and report.invalid_coordinates == 2


def test_ratings_above_five_are_missing():
    frame, report = clean(_row(Delivery_person_Ratings="6"))
    assert pd.isna(frame.loc[0, "courier_rating"]) and report.ratings_out_of_range == 1


def test_haversine_known_distance():
    assert haversine_km(12.97, 77.59, 13.08, 80.27) == pytest.approx(290.0, rel=0.02)
    assert haversine_km(1, 1, 1, 1) == 0.0


def test_courier_id_is_not_a_feature_but_its_city_is():
    assert courier_city(pd.Series(["INDORES13DEL02", "RANCHIRES09DEL01", "bad"])).tolist() == ["INDO", "RANCHI", "UNKNOWN"]
    for ident in EXCLUDED:
        assert ident not in FEATURES
    assert "courier_city" in CATEGORICAL


def test_interaction_feature_is_a_model_input(frame):
    assert "traffic_x_distance" in NUMERIC
    row = frame.dropna(subset=["traffic_level", "distance_km"]).iloc[0]
    assert row["traffic_x_distance"] == pytest.approx(row["traffic_level"] * row["distance_km"])


def test_missing_column_is_refused():
    with pytest.raises(DataQualityError, match="missing columns"):
        clean(_row().drop(columns=["City"]))
    frame, _ = clean(_row().drop(columns=["Time_taken(min)"]), require_target=False)
    assert "time_taken_min" not in frame
