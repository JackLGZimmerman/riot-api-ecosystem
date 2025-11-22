# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps here if needed
# RUN apt-get update && apt-get install -y ...

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv pip install --system .

COPY . .

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]