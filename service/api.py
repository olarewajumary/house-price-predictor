from fastapi import FastAPI, HTTPException

from service.predict import encoders, predict_price
from service.schemas import HouseFeatures

app = FastAPI(title="House Price API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    if features.neighborhood not in list(encoders["Neighborhood"].classes_):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown neighborhood: {features.neighborhood}",
        )
    if features.exter_qual not in list(encoders["ExterQual"].classes_):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown exterior quality: {features.exter_qual}",
        )

    price = predict_price(**features.model_dump())
    return {"predicted_price": round(price, 2)}