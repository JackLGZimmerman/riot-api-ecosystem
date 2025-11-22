from celery import Celery  # type: ignore[import]

celery_app = Celery("data_ecosystem")

# Pass the module path string; no import of "settings" required
celery_app.config_from_object("app.workers.config")  # type: ignore[reportUnknownMemberType]
celery_app.autodiscover_tasks(["app.workers.tasks"])  # type: ignore[reportUnknownMemberType]
