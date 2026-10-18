import json

import httpx

URL = "http://127.0.0.1:8000/predict"

with open("sample.json") as f:
    good = json.load(f)
with open("sample_bad.json") as f:
    bad = json.load(f)

with httpx.Client(timeout=30) as client:
    for _ in range(100):
        client.post(URL, json=good)
    for _ in range(20):
        client.post(URL, json=bad)

print("done: sent 100 good and 20 bad requests")