# app/services/utils/logging/logger.py

from __future__ import annotations

import json
import logging.config
import logging.handlers
from pathlib import Path


def setup_logging_config() -> None:
    config_path = Path(__file__).resolve().parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    log_file = Path(config["handlers"]["json_file"]["filename"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)
