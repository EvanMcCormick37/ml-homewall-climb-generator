FROM python:3.11-slim AS base

WORKDIR /app

# Install python system dependencies which require C++ Compiler during install
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ build-essential && rm -rf /var/lib/apt/lists/*

# Install python dependencies from wheel
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy app and Data directories
COPY app/ ./app/
COPY data/models/ ./data/models/

RUN mkdir -p ./data/walls

EXPOSE 8000

# Health check — Docker (and platforms like Railway/Fly) use this to know
# the container is alive and ready to serve traffic
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the app with uvicorn
# --host 0.0.0.0: listen on all interfaces (required inside a container)
# --workers 2: run 2 worker processes (good for a 2-core VM; adjust to your deployment)
# --timeout-keep-alive 65: slightly above typical load balancer timeout (60s)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "65"]