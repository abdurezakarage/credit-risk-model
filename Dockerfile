# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Install Python packages
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy source code
COPY ./src ./src

# Default command to run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
