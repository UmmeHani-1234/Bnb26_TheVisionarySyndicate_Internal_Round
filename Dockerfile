# Production Dockerfile for Black Box AI Agent Observability & Debugging System
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    API_PORT=8000

# Install minimal OS dependencies for health checks and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Streamlit UI port (8501) and FastAPI API port (8000)
EXPOSE 8501 8000

# Health check against FastAPI backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint starts both FastAPI and Streamlit concurrently
CMD ["python", "run_production.py"]
