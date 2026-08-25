FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; pypdf/boto3/qdrant-client are pure-Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Local metadata registry lives under /app/data inside the container;
# mount a volume here in docker-compose to persist it across restarts.
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1 \
    DOCUMENTS_REGISTRY_PATH=/app/data/documents_registry.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
