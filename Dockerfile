FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
 && uv sync --frozen

CMD ["/app/.venv/bin/python", "-m", "app.worker.pipelines.recurring_runner"]