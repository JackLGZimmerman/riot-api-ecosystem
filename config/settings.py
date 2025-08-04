from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_key: str
    calls_per_two_minutes: int = 100
    time_period_two_minutes: float = 120.0
    league_page_upper_bound: int = 1024
    base_project_path: str

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()

if __name__ == "__main__":
    print(settings.model_dump())