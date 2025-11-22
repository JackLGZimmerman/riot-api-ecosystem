# app/api/v1/routes/pipelines.py
from fastapi import APIRouter, status

from app.workers.tasks.pipeline import player_collection_task

router = APIRouter(
    prefix="/pipelines",
    tags=["pipelines"],
)


@router.post(
    "/player_collection",
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_player_collection():
    """
    Trigger the league snapshot pipeline as a background Celery task.

    Returns immediately with the Celery task_id so the caller
    can track it if needed.
    """
    async_result = player_collection_task.delay()  # type: ignore[attr-defined]

    return {
        "task_id": async_result.id,
        "status": "queued",
    }
