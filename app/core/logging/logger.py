from __future__ import annotations

import json
import logging.config
import logging.handlers
from pathlib import Path
from typing import Any


def _resolve_log_path(filename: str, project_root: Path) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return project_root / path


def _ensure_file_handler_dirs(config: dict[str, Any], *, project_root: Path) -> None:
    handlers = config.get("handlers", {})
    for handler_cfg in handlers.values():
        filename = handler_cfg.get("filename")
        if not filename:
            continue

        log_path = _resolve_log_path(filename, project_root)
        handler_cfg["filename"] = str(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging_config() -> None:
    config_path = Path(__file__).resolve().parent / "config.json"
    project_root = config_path.parents[3]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    _ensure_file_handler_dirs(config, project_root=project_root)

    logging.config.dictConfig(config)
