# app/core/config/settings.py

from __future__ import annotations

from pathlib import Path

from pydantic import PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    api_key: SecretStr

    rate_limit_calls: PositiveInt = 100
    rate_limit_period: PositiveInt = 120
    rate_limiter_debug: bool = False

    base_project_path: Path = PROJECT_ROOT

    clickhouse_host: str
    clickhouse_port: PositiveInt = 8123
    clickhouse_database: str
    clickhouse_user: str
    clickhouse_password: SecretStr
    clickhouse_send_receive_timeout: PositiveInt = 1800

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]
