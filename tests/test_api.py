from fastapi.testclient import TestClient

from service.api import app

client = TestClient(app)


def test_predict_returns_price():
    payload = {
        "overall_qual": 7,
        "gr_liv_area": 1800,
        "total_bsmt_sf": 1000,
        "garage_cars": 2,
        "year_built": 2005,
        "full_bath": 2,
        "neighborhood": "NAmes",
        "exter_qual": "TA",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    price = response.json()["predicted_price"]
    assert 173000 < price < 174000


def test_predict_rejects_bad_input():
    payload = {
        "overall_qual": 11,
        "gr_liv_area": 1800,
        "total_bsmt_sf": 1000,
        "garage_cars": 2,
        "year_built": 2005,
        "full_bath": 2,
        "neighborhood": "NAmes",
        "exter_qual": "TA",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422    