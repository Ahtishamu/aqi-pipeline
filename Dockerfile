# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    STREAMLIT_PORT=8501

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
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] streamlit requests

# Copy application code
COPY model_training/ ./model_training/
COPY ci_cd/ ./ci_cd/
COPY backend/ ./backend/
COPY app/ ./app/
COPY *.py ./

# Create directories for outputs
RUN mkdir -p model_results logs

# Create non-root user for security
RUN useradd -m -u 1000 aqi_user && \
    chown -R aqi_user:aqi_user /app
USER aqi_user

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Simple start script: launch FastAPI then Streamlit
CMD bash -c "uvicorn backend.main:app --host 0.0.0.0 --port $PORT & \n                streamlit run app/main_app.py --server.address 0.0.0.0 --server.port $STREAMLIT_PORT"

# Labels for metadata
LABEL name="aqi-pipeline" \
      version="1.1" \
      description="AQI prediction pipeline with API and dashboard" \
      maintainer="ahtishamu"
