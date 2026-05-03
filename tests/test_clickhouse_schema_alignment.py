from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "database" / "clickhouse" / "schema"
sys.path.insert(0, str(REPO_ROOT))


def _schema_columns(schema_name: str) -> set[str]:
    columns: set[str] = set()
    schema_path = SCHEMA_DIR / schema_name

    for line in schema_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if stripped.startswith(
            ("CREATE TABLE", "(", ")", "ENGINE", "ORDER BY", "SETTINGS")
        ):
            continue

        columns.add(stripped.split()[0].rstrip(","))

    return columns


def _timeline_participant_stats_model_columns() -> set[str]:
    from app.services.riot_api_client.parsers.models.timeline import (
        ChampionStats,
        DamageStats,
    )

    return {
        "run_id",
        "matchid",
        "frame_timestamp",
        "participantid",
        *(field.lower() for field in ChampionStats.model_fields),
        "currentgold",
        *(field.lower() for field in DamageStats.model_fields),
        "goldpersecond",
        "jungleminionskilled",
        "level",
        "minionskilled",
        "position_x",
        "position_y",
        "timeenemyspentcontrolled",
        "totalgold",
        "xp",
    }


def _participant_challenges_model_columns() -> set[str]:
    from app.services.riot_api_client.parsers.models.non_timeline import Challenges

    return {
        "run_id",
        "matchid",
        "teamid",
        "puuid",
        *(field.lower() for field in Challenges.model_fields),
    }


def test_tl_participant_stats_model_matches_clickhouse_schema() -> None:
    assert _timeline_participant_stats_model_columns() == _schema_columns(
        "3103_tl_participant_stats_schema.sql"
    )


def test_participant_challenges_model_matches_clickhouse_schema() -> None:
    assert _participant_challenges_model_columns() == _schema_columns(
        "3116_participant_challenges_schema.sql"
    )
