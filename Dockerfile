# ===== FILE: Dockerfile =====
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Disable heavy downloads during Docker build
ENV SKIP_MODEL_DOWNLOAD=true

EXPOSE 5000

CMD ["python", "app.py"]
