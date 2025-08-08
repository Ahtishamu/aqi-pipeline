# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY model_training_requirements.txt .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r model_training_requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_training/ ./model_training/
COPY ci_cd/ ./ci_cd/
COPY *.py ./

# Create directories for outputs
RUN mkdir -p model_results logs

# Create non-root user for security
RUN useradd -m -u 1000 aqi_user && \
    chown -R aqi_user:aqi_user /app
USER aqi_user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import hopsworks; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "update_features_hourly.py"]

# Labels for metadata
LABEL name="aqi-pipeline" \
      version="1.0" \
      description="AQI prediction and monitoring pipeline" \
      maintainer="ahtishamu"
