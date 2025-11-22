# app/api/v1/routes/tasks.py

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models import BasicBoundsConfig, EliteBoundsConfig
from app.services import stream_elite_players, stream_sub_elite_players
from app.workers.tasks.pipeline import long_running_task

router = APIRouter()


@router.post("/run-long-task")
def run_long_task(n: int):
    result = long_running_task.delay(n)  # type: ignore[attr-defined]
    return {"task_id": result.id}


@router.post("/stream-elite-players")
def stream_elite_players_task(bounds: EliteBoundsConfig):
    async def gen():
        async for entry in stream_elite_players(bounds):
            yield entry.model_dump_json() + "\n"

    return StreamingResponse(gen(), media_type="application/json")


@router.post("/stream-sub-elite-players")
def stream_sub_elite_players_task(bounds: BasicBoundsConfig):
    async def gen():
        async for entry in stream_sub_elite_players(bounds):
            yield entry.model_dump_json() + "\n"

    return StreamingResponse(gen(), media_type="application/json")



