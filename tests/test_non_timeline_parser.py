from __future__ import annotations

import copy
import json
from pathlib import Path

from app.services.riot_api_client.parsers.non_timeline import (
    MatchDataNonTimelineParsingOrchestrator,
)


def test_non_timeline_accepts_matchmaking_role_drift_fields() -> None:
    raw = json.loads(Path("non-timeline.example.json").read_text(encoding="utf-8"))
    raw["metadata"]["matchId"] = "EUW1_7869991463"

    for participant in raw["info"]["participants"]:
        participant["positionAssignedByMatchmaking"] = "TOP"
        participant["selectedRolePreferences"] = "TOP.PRIMARY.TOP.JUNGLE"

    tables = MatchDataNonTimelineParsingOrchestrator().run(copy.deepcopy(raw))

    assert tables.participant_stats[0]["positionAssignedByMatchmaking"] == "TOP"
    assert (
        tables.participant_stats[0]["selectedRolePreferences"]
        == "TOP.PRIMARY.TOP.JUNGLE"
    )
