# HayatKurtaranAI — Dockerfile
# Multi-stage build for minimal image size
FROM python:3.12-slim AS base

# Metadata
LABEL maintainer="HayatKurtaranAI Team"
LABEL description="Turkish First-Aid RAG Chatbot with Emergency Triage"
LABEL version="2.0"

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-build FAISS cache (speeds up first startup)
RUN python -c "from backend.vector_db import initialize; initialize()" || true

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
