from service.predict import predict_price


def test_sample_house_price():
    price = predict_price(7, 1800, 1000, 2, 2005, 2, "NAmes", "TA")
    assert 1 < price < 2