from fastapi import FastAPI

from service.predict import predict_price
from service.schemas import HouseFeatures

app = FastAPI(title="House Price API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    price = predict_price(**features.model_dump())
    return {"predicted_price": round(price, 2)}