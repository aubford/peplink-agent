FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout 1000 -r requirements-prod.txt

# Create data directories (do this before copying code)
RUN mkdir -p data logs

# Copy the application code (this should be last to maximize cache hits)
COPY . .

WORKDIR /app/web_app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/threads || exit 1

# Run the application
CMD ["uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
