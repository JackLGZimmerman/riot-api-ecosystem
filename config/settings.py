from __future__ import annotations
from pathlib import Path
from pydantic import SecretStr, PositiveInt, PositiveFloat
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- external/api ---
    api_key: SecretStr
    rate_limit_calls: PositiveInt = 100
    rate_limit_period: PositiveFloat = 120.0
    base_project_path: Path = Path.cwd()

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]
