# Build from the repository root:
#   docker build -f src/backend/Dockerfile -t drugresponse-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
COPY src/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ai_model /app/src/ai_model
COPY src/backend /app/src/backend
COPY src/frontend /app/src/frontend

RUN useradd --create-home appuser && mkdir -p /app/models /app/storage && chown -R appuser /app
USER appuser

ENV MODEL_DIR=/app/models/latest STORAGE_DIR=/app/storage FRONTEND_DIR=/app/src/frontend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
