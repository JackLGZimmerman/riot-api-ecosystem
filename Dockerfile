FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
 && uv sync --frozen --no-dev

# Prefect executes `python` inside this container; point PATH at uv's env.
ENV PATH="/app/.venv/bin:${PATH}"

COPY . .
