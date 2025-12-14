# Base Dockerfile for GIL Saturation Cliff Experiments.
# Simulates an edge device environment with CPU constraints.

FROM python:3.10-slim

LABEL maintainer="Mridankan Mandal"
LABEL description="GIL Saturation Cliff Benchmark Environment"
LABEL version="1.0"

# Set working directory.
WORKDIR /app

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code.
COPY src/ ./src/
COPY experiments/ ./experiments/

# Create results directory.
RUN mkdir -p results figures

# Set Python path.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command runs the single-core benchmark.
CMD ["python", "experiments/singleCoreBenchmark.py"]
