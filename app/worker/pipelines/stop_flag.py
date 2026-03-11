from __future__ import annotations

import os
from pathlib import Path

STOP_FLAG_ENV_VAR = "PIPELINE_STOP_FLAG_PATH"
DEFAULT_STOP_FLAG_PATH = "/tmp/riot_pipeline_stop_requested"


def get_stop_flag_path() -> Path:
    return Path(os.getenv(STOP_FLAG_ENV_VAR, DEFAULT_STOP_FLAG_PATH))


def raise_if_stop_requested(*, stage: str) -> None:
    flag_path = get_stop_flag_path()
    if flag_path.exists():
        raise RuntimeError(f"Stop requested via flag file: {flag_path} (stage={stage})")
