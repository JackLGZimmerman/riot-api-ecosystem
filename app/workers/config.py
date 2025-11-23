# app/workers/config.py

import os
from datetime import timedelta

from app.core.config.settings import settings

broker_url = os.getenv("BROKER_URL", "amqp://user:password@localhost:5672//")
result_backend = None

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

task_track_started = True
worker_prefetch_multiplier = 1
task_acks_late = True
task_default_queue = "default"

# -----------------------------
# PERIODIC TASK CONFIG (BEAT)
# -----------------------------

beat_schedule = {
    "player_collection_every_n_days": {
        "task": "pipelines.player_collection",
        "schedule": timedelta(days=settings.data_collection_frequency),
        "options": {
            "queue": "default",
        },
    },
}
