from __future__ import annotations
from pathlib import Path
from typing import List
from pydantic import SecretStr, PositiveInt, PositiveFloat
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- external/api ---
    api_key: SecretStr
    calls_per_two_minutes: PositiveInt = 100
    time_period_two_minutes: PositiveFloat = 120.0
    league_page_upper_bound: PositiveInt = 1024
    base_project_path: Path = Path.cwd()

    # --- database ---
    puuid_ttl_days: PositiveInt = 90
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

if __name__ == "__main__":
    print(settings.model_dump())
