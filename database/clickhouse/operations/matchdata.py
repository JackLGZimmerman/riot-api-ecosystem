from typing import Iterable
from uuid import UUID

from database.clickhouse.client import get_client


def delete_by_run_id_and_matchids(
    table: str,
    run_id: UUID,
    match_ids: Iterable[str],
) -> None:
    ids = list(dict.fromkeys(str(match_id) for match_id in match_ids if match_id))
    if not ids:
        return
    numeric_ids = _extract_numeric_match_ids(ids)

    # RECOVERY-SYSTEM: per-batch rollback for failed match IDs only.
    command = f"""
        ALTER TABLE {table}
        DELETE
        WHERE run_id = %(run_id)s
          AND (
              has(%(match_ids)s, toString(matchid))
              OR has(%(numeric_match_ids)s, toString(matchid))
          )
    """
    get_client().command(
        command,
        parameters={
            "run_id": str(run_id),
            "match_ids": ids,
            "numeric_match_ids": numeric_ids,
        },
    )


def _extract_numeric_match_ids(match_ids: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match_id in match_ids:
        candidate = match_id.rsplit("_", 1)[-1]
        if not candidate.isdigit() or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out
