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


## Production API (FastAPI, Docker, CI)

The Streamlit demo above is the front end. This section is the engineering layer around the same trained model: a validated API, automated tests, a container, and a CI pipeline.

**What it does**

- `POST /predict` takes the 8 house features and returns a predicted sale price.
- `GET /health` returns `{"status": "ok"}` for uptime checks.
- Pydantic validates every request. Out-of-range values and unknown neighborhoods return a clear `422` error instead of a crash.
- Every request is logged as one line of JSON (method, path, status, latency).
- The artifacts folder is configurable through the `ARTIFACTS_DIR` environment variable.

**Run it locally**

```
pip install -r requirements-api.txt
uvicorn service.api:app
```

Then open http://127.0.0.1:8000/docs for the interactive API page.

**Run it in Docker**

```
docker build --load -t house-price-api .
docker run -p 8000:8000 house-price-api
```

**Tests and CI**

`pytest` runs 4 tests: one on the prediction function and three through the API, including bad-input cases. GitHub Actions runs lint (ruff), the tests, and a Docker build on every push. The Actions history includes a deliberately broken commit that failed the pipeline, and the fix that turned it green again.

**Measured reliability**

From a load test of 120 requests (100 valid, 20 invalid) on a local Windows laptop, sent one after another:

| Metric | Result |
| --- | --- |
| Server errors (5xx) | 0 |
| Invalid requests rejected cleanly (422) | 20 of 20 |
| Median latency (`/predict`) | 91 ms |
| p95 latency (`/predict`) | 156 ms |

This is a local, single-user measurement, not a production load test. The load script is `scripts/load_test.py` and the report is `scripts/report.py`.