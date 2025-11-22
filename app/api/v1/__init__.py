from fastapi import APIRouter

from app.api.v1.routes import tasks
from app.api.v1.routes import pipelines

router = APIRouter()
router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
