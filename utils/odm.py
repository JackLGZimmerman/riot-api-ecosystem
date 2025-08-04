from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Any, AsyncGenerator, TypeVar, Type
from config import db_settings

from beanie import Document, init_beanie
from pydantic import Field

from pymongo import IndexModel, AsyncMongoClient, UpdateOne
from pymongo.errors import BulkWriteError


T = TypeVar("T", bound=Document)
MAX_PUUIDS_PER_BUCKET = 10_000


class Player(Document):
    puuid: str          = Field(..., description="Unique player identifier")
    queueType: str      = Field(..., description="The queue the game was played in")
    tier: str           = Field(..., description="The tier (e.g., DIAMOND)")
    rank: str           = Field(..., description="The rank (e.g. I, II, IV)")
    wins: int           = Field(..., ge=0, description="The games the player has won")
    losses: int         = Field(..., ge=0, description="The games the player has lost")
    region: str         = Field(..., description="The region the player currently resides in")
    continent: str      = Field(..., description="The continent in which the region resides in")

    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),description="For TTL")

    class Settings:
        name = "players"
        indexes = [
            IndexModel([("puuid", 1), ("queueType", 1)], unique=True),
            IndexModel([("createdAt", 1)], expireAfterSeconds=db_settings.puuid_ttl_days * 24 * 60 * 60)
        ]
    
    @classmethod
    async def bulk_upsert(cls, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            return

        batch_ts        = datetime.now(timezone.utc)
        updatable_keys  = [
            k 
            for k in cls.model_fields
            if k not in ("puuid", "queueType", "createdAt")
        ]

        ops: List[UpdateOne] = []
        for idx, doc in enumerate(docs, start=1):
            try:
                set_fields = {k: doc[k] for k in updatable_keys if k in doc}
                on_insert = {"createdAt": batch_ts}
                ops.append(UpdateOne(
                    {
                        "puuid": doc["puuid"],
                        "queueType": doc["queueType"]
                    },
                    {
                        "$set": set_fields, 
                        "$setOnInsert": on_insert
                    }, upsert=True))
            except Exception as e:
                print(f"  ✖ Error preparing op #{idx}): {e}")

        try:
            await cls.get_pymongo_collection().bulk_write(
                ops=ops,
                ordered=False,
                bypass_document_validation=True
            )
        except BulkWriteError as bwe:
            print("BulkWriteError:", bwe.details)
        except Exception as e:
            print("Error during bulk_write():", e)

    @classmethod
    async def get_puuid_continent_queuetype():
        cursor = Player.find({})
        async for player in cursor:
            yield {
                player.puuid, 
                player.continent, 
                player.queueType
            }


DOCUMENT_MODELS = [Player]
_client: AsyncMongoClient | None = None

async def init_db() -> None:
    global _client
    if not _client:
        _client = AsyncMongoClient(db_settings.mongo_uri)


    db = _client[db_settings.mongo_database]
    await init_beanie(database=db, document_models=DOCUMENT_MODELS)
    return _client

async def close_db() -> None:
    global _client
    if _client:
        await _client.close()
        _client = None