FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.inference.txt /app/requirements.inference.txt
RUN pip install --no-cache-dir -r /app/requirements.inference.txt

COPY src /app/src
COPY mlops /app/mlops
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml

ENTRYPOINT ["python", "src/mlops_batch.py"]
