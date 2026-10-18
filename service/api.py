import json
import logging
import time

from fastapi import FastAPI, HTTPException, Request

from service.predict import encoders, predict_price
from service.schemas import HouseFeatures

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("house_api")

app = FastAPI(title="House Price API")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": latency_ms,
                }
            )
        )


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