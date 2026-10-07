# data/

Git ignores every file in this folder except this README. Do not commit the Kaggle files.
The offline demo and the tests do not need this folder: `eta-eats synth --out data` writes synthetic
`train.csv`, `test.csv` and `Sample_Submission.csv` in the same raw format, with the same quirks.

## Expected files

| File | Rows (Kaggle) | Contents |
|---|---|---|
| `train.csv` | 45,593 | 19 input columns and the target `Time_taken(min)` (text such as `(min) 24`) |
| `test.csv` | about 11,400 | The 19 input columns, no target |
| `Sample_Submission.csv` | about 11,400 | `ID`, `Time_taken (min)`: the submission format |

Input columns: `ID`, `Delivery_person_ID`, `Delivery_person_Age`, `Delivery_person_Ratings`, `Restaurant_latitude`,
`Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`, `Order_Date` (`dd-mm-yyyy`),
`Time_Orderd`, `Time_Order_picked` (`HH:MM:SS`), `Weatherconditions`, `Road_traffic_density`, `Vehicle_condition`,
`Type_of_order`, `Type_of_vehicle`, `multiple_deliveries`, `Festival`, `City`.

Known values (after the loader removes spaces and the `conditions` prefix):

| Column | Values |
|---|---|
| `Road_traffic_density` | `Low`, `Medium`, `High`, `Jam` |
| `Weatherconditions` | `Sunny`, `Cloudy`, `Fog`, `Windy`, `Stormy`, `Sandstorms` |
| `Type_of_order` | `Snack`, `Meal`, `Drinks`, `Buffet` |
| `Type_of_vehicle` | `motorcycle`, `scooter`, `electric_scooter`, `bicycle` |
| `Festival` | `No`, `Yes` |
| `City` | `Urban`, `Semi-Urban`, `Metropolitian` (spelling as in the source) |

The string `NaN` means a missing value. Any other unknown value stops the loader.

## Source

| Item | Value |
|---|---|
| Name | Food Delivery Dataset (Kaggle, by gauravmalik26) |
| URL | https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset |
| License | Read the dataset page before you use or share the files |
| Period | Orders from 11 February 2022 to 6 April 2022, Indian cities |

## How to download

1. Download the three CSV files from the Kaggle page.
2. Save them in `data/` with their original names.
3. Run `eta-eats clean --data data/train.csv` and read the data quality report.
