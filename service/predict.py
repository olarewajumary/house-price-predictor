from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Loaded once, when this file is first imported
model = joblib.load(ARTIFACTS_DIR / "model.pkl")
encoders = joblib.load(ARTIFACTS_DIR / "encoders.pkl")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
defaults = joblib.load(ARTIFACTS_DIR / "defaults.pkl")
columns = joblib.load(ARTIFACTS_DIR / "columns.pkl")



def predict_price(
    overall_qual,
    gr_liv_area,
    total_bsmt_sf,
    garage_cars,
    year_built,
    full_bath,
    neighborhood,
    exter_qual,
) -> float:
    row = defaults.copy()
    row["OverallQual"] = overall_qual
    row["GrLivArea"] = gr_liv_area
    row["TotalBsmtSF"] = total_bsmt_sf
    row["GarageCars"] = garage_cars
    row["YearBuilt"] = year_built
    row["FullBath"] = full_bath
    row["Neighborhood"] = neighborhood
    row["ExterQual"] = exter_qual

    input_df = pd.DataFrame([row])[columns]

    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col].astype(str))

    numeric_cols = [c for c in input_df.columns if c in scaler.feature_names_in_]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    log_prediction = model.predict(input_df)[0]
    return float(np.expm1(log_prediction))