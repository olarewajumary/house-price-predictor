FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY service/ service/
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn service.api:app --host 0.0.0.0 --port ${PORT:-8000}"]