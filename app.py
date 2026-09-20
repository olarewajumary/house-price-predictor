import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("artifacts/model.pkl")
encoders = joblib.load("artifacts/encoders.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
defaults = joblib.load("artifacts/defaults.pkl")
columns = joblib.load("artifacts/columns.pkl")

st.title("House Price Predictor")
st.write("Estimate a home's sale price using a Ridge Regression model trained on the Ames Housing dataset.")

overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 6000, 1000)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
year_built = st.number_input("Year Built", 1870, 2026, 2005)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
neighborhood = st.selectbox("Neighborhood", list(encoders['Neighborhood'].classes_))
exter_qual = st.selectbox("Exterior Quality", list(encoders['ExterQual'].classes_))

if st.button("Predict Price"):
    try:
        row = defaults.copy()

        row['OverallQual'] = overall_qual
        row['GrLivArea'] = gr_liv_area
        row['TotalBsmtSF'] = total_bsmt_sf
        row['GarageCars'] = garage_cars
        row['YearBuilt'] = year_built
        row['FullBath'] = full_bath
        row['Neighborhood'] = neighborhood
        row['ExterQual'] = exter_qual

        input_df = pd.DataFrame([row])
        input_df = input_df[columns]

        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col].astype(str))

        numeric_cols = [c for c in input_df.columns if c in scaler.feature_names_in_]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        log_prediction = model.predict(input_df)[0]
        price = np.expm1(log_prediction)

        st.success(f"Estimated Sale Price: ${price:,.0f}")

    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")