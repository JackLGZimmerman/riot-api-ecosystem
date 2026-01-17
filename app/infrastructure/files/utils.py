import logging
import os
import shutil
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

logger = logging.getLogger(__name__)


@asynccontextmanager
async def atomic_outputs(*final_paths: Path) -> AsyncIterator[tuple[Path, ...]]:
    tmp_paths: list[Path] = []
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []

    def _remove_path(p: Path) -> None:
        if not p.exists():
            return
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)

    def clear_tmp_paths_if_exist() -> None:
        for final in final_paths:
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp = final.with_name(f".{final.name}.{uuid4().hex}.tmp")
            _remove_path(tmp)
            tmp_paths.append(tmp)

    clear_tmp_paths_if_exist()

    try:
        yield tuple(tmp_paths)

        # User has done work with the tmp_paths
        for final, tmp in zip(final_paths, tmp_paths):
            # Some reason those tmp files don't exist anymore
            if not tmp.exists():
                raise FileNotFoundError(f"Temp output missing: {tmp}")

            # If the original filenames the user provided still exist we create a back-up label, and check if that back-up label already exists
            # We then replace the original user filename with the back-up name and append these paths to the back-up list
            if final.exists():
                backup = final.with_name(f".{final.name}.{uuid4().hex}.bak")
                _remove_path(backup)
                os.replace(final, backup)
                backups.append((final, backup))

            # We replace the tmp filename with the original users filename, and officially commit that change
            os.replace(tmp, final)
            committed.append(final)

        # if everything went well we can safely delete the back-ups
        for _, backup in backups:
            _remove_path(backup)

    except Exception:
        # Remove all of the data from tmp, back-ups and comitted without raising exceptions, and THEN raise.
        for tmp in tmp_paths:
            with suppress(Exception):
                _remove_path(tmp)

        for final in committed:
            with suppress(Exception):
                _remove_path(final)

        for final, backup in backups:
            with suppress(Exception):
                if backup.exists():
                    os.replace(backup, final)

        raise
