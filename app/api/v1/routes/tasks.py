# app/api/v1/routes/tasks.py

from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from app.models import BasicBoundsConfig, EliteBoundsConfig
from app.services import stream_elite_players, stream_sub_elite_players
from app.services.riot_api_client.base import get_riot_api
from app.workers.tasks.pipeline import long_running_task
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
router = APIRouter()


@router.post("/run-long-task")
def run_long_task(n: int):
    result = long_running_task.delay(n)  # type: ignore[attr-defined]
    return {"task_id": result.id}


@router.post("/stream-elite-players")
def stream_elite_players_task(bounds: EliteBoundsConfig):
    async def gen():
        async with get_riot_api() as riot_api:
            async for entry in stream_elite_players(bounds, riot_api):
                yield entry.model_dump_json() + "\n"

    return StreamingResponse(gen(), media_type="application/json")


@router.post("/stream-sub-elite-players")
def stream_sub_elite_players_task(bounds: BasicBoundsConfig):
    async def gen():
        async with get_riot_api() as riot_api:
            async for entry in stream_sub_elite_players(bounds, riot_api):
                yield entry.model_dump_json() + "\n"

    return StreamingResponse(gen(), media_type="application/json")

@router.get("/metrics")
def get_app_metrics():
    data = generate_latest()  # bytes with Prometheus exposition format
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)