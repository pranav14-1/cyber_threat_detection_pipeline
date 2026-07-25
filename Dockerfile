# ==============================================================================
# Base Image
# ==============================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install basic system dependencies required for compilation or downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Install Dependencies (Cached Layer)
# ==============================================================================
# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Set Up App User
# ==============================================================================
# Create a non-privileged user to run the app securely
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser && \
    chown -R appuser:appgroup /app

# ==============================================================================
# Copy Source & Set Environment
# ==============================================================================
# Switch to the non-root user
USER appuser

# Copy the rest of the application files
COPY --chown=appuser:appgroup . .

# Create data directories inside container with correct permissions
RUN mkdir -p data/raw data/processed models

# Expose port (Streamlit default)
EXPOSE 8501

# Healthcheck to verify the web service is running and responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit dashboard by default
CMD ["streamlit", "run", "src/phase6_dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
