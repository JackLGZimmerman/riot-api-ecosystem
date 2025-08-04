import json
from pathlib import Path
from typing import Annotated, List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode

class DatabaseSettings(BaseSettings):
    mongo_uri: str
    mongo_database: str
    mongo_collections: Annotated[List[str], NoDecode]
    puuid_ttl_days: int

    mongo_database_ftl: str
    mongo_collection_ftl_players: str
    mongo_collection_ftl_puuids: str

    @field_validator("mongo_collections", mode="before")
    @classmethod
    def _parse_collections(cls, v: Union[str, List[str]]):
        """
        Allow either:
          - JSON array:    ["players","teams"]
          - CSV string:    players,teams
        """
        if isinstance(v, list):
            return v
        try:
            return json.loads(v)
        except Exception:
            return [item.strip() for item in v.split(",") if item.strip()]

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        case_sensitive=False,
        extra="ignore",
    )

db_settings = DatabaseSettings()

if __name__ == "__main__":
    print(db_settings.model_dump())