from fastapi import FastAPI

app = FastAPI(title="House Price API")


@app.get("/health")
def health():
    return {"status": "ok"}