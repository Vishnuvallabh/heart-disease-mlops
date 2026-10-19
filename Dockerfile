# Small official Python image, same Python version as the venv and CI
FROM python:3.12-slim

# No .pyc files, logs printed immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first: this layer is cached until requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Then copy only what the API needs
COPY src/ src/
COPY models/ models/

# Run as a non-root user (safer in production)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
