# House Price Predictor

Predicts what a house would sell for based on things like square footage, quality rating, and neighborhood, trained on the Ames Housing dataset (the same one Kaggle uses for its housing competition).

**Live demo:** https://house-price-predictor-inyunbm8dhh25ppfbpnjci.streamlit.app/

![App screenshot](screenshots/demo.png)

## What's going on under the hood

The dataset has 80+ columns and a good number of missing values scattered across it, so most of the actual work here is cleaning: filling in gaps (median for numbers, most common value for categories, and lot frontage specifically gets filled per-neighborhood since lot sizes vary a lot by area), encoding categorical columns, and scaling everything numeric.

**Model:** Ridge Regression. I chose Ridge over plain Linear Regression because the dataset has a lot of correlated features (like total square footage and above-ground living area), and Ridge's regularization keeps the model from overreacting to that overlap. The `alpha` value is tuned automatically with `GridSearchCV` (testing 0.1 through 100, using 5-fold cross-validation) rather than picked by hand, it landed on 100.

One thing that made a real difference: house prices are skewed, so a few very expensive homes were throwing off the error metric more than they should. Training on the log of the price instead of the raw number, then converting predictions back afterward, fixed that.

Ends up at an RMSE of about $27,900 and an R² of 0.899.

## Why the demo only asks for 8 things

Nobody's filling out an 80-field form to get a price estimate. The app asks for the 8 features that actually move the needle such as quality, size, garage, year built, bathrooms, neighborhood, exterior quality and quietly fills in the rest using the same defaults the model was trained with.

## Running it yourself
```
git clone https://github.com/olarewajumary/house-price-predictor
cd house-price-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
streamlit run app.py
```

`main.py` trains the model and saves it; `app.py` is the actual interface.

## Built with

Python, pandas, scikit-learn, Streamlit