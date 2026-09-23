FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV MODEL_URI=/app/release/model
ENV MODEL_MANIFEST_PATH=/app/release/model-manifest.json

COPY requirements-runtime.txt ./

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --no-cache-dir --no-deps .

COPY release ./release

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "satellite_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
