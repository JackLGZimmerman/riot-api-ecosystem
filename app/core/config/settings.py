# app/core/config/settings.py

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    api_key: SecretStr

    rate_limit_calls: PositiveInt = 100
    rate_limit_period: PositiveInt = 120
    rate_limit_burst_calls: PositiveInt = 20
    rate_limit_burst_period: PositiveInt = 1

    base_project_path: Path = PROJECT_ROOT

    clickhouse_host: str
    clickhouse_port: PositiveInt = 8123
    clickhouse_database: str
    clickhouse_user: str
    clickhouse_password: SecretStr

    threadpool_executor_clickhouse: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="clickhouse"
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]
