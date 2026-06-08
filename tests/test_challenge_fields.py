"""Golden test pinning the participant-challenges field contract.

The ~133 challenge fields are defined once as a canonical ordered tuple and used
to generate both the `Challenges` Pydantic model (drift detection) and the
`TabulatedParticipantChallenges` output TypedDict (ClickHouse column order).
This test pins the exact, runtime-load-bearing facts so the de-duplication
cannot reorder wire columns or drop the `12AssistStreakCount` alias /
`legendaryItemUsed: list[int]` special cases.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.services.riot_api_client.parsers.models.non_timeline import Challenges
from app.services.riot_api_client.parsers.non_timeline import (
    ParticipantChallengesParser,
    TabulatedParticipantChallenges,
    _CHALLENGE_NUMERIC_FIELDS,
)
from app.worker.pipelines.matchdata_orchestrator import columns_from_typed_dict

# The exact, ordered set of challenge field names (wire-column order).
EXPECTED_CHALLENGE_FIELDS: tuple[str, ...] = (
    "x12AssistStreakCount", "HealFromMapSources", "InfernalScalePickup", "abilityUses",
    "acesBefore15Minutes", "alliedJungleMonsterKills",
    "baronBuffGoldAdvantageOverThreshold", "baronTakedowns",
    "blastConeOppositeOpponentCount", "bountyGold", "buffsStolen",
    "completeSupportQuestInTime", "controlWardTimeCoverageInRiverOrEnemyHalf",
    "controlWardsPlaced", "damagePerMinute", "damageTakenOnTeamPercentage",
    "dancedWithRiftHerald", "deathsByEnemyChamps", "dodgeSkillShotsSmallWindow",
    "doubleAces", "dragonTakedowns", "earliestBaron", "earliestDragonTakedown",
    "earliestElderDragon", "earlyLaningPhaseGoldExpAdvantage",
    "effectiveHealAndShielding", "elderDragonKillsWithOpposingSoul",
    "elderDragonMultikills", "enemyChampionImmobilizations", "enemyJungleMonsterKills",
    "epicMonsterKillsNearEnemyJungler", "epicMonsterKillsWithin30SecondsOfSpawn",
    "epicMonsterSteals", "epicMonsterStolenWithoutSmite", "firstTurretKilled",
    "firstTurretKilledTime", "fasterSupportQuestCompletion", "fastestLegendary",
    "fistBumpParticipation", "flawlessAces", "fullTeamTakedown", "gameLength",
    "getTakedownsInAllLanesEarlyJungleAsLaner", "goldPerMinute", "hadOpenNexus",
    "hadAfkTeammate", "highestChampionDamage", "highestCrowdControlScore",
    "highestWardKills", "immobilizeAndKillWithAlly", "initialBuffCount",
    "initialCrabCount", "jungleCsBefore10Minutes", "junglerKillsEarlyJungle",
    "junglerTakedownsNearDamagedEpicMonster", "kTurretsDestroyedBeforePlatesFall",
    "kda", "killAfterHiddenWithAlly", "killParticipation",
    "killedChampTookFullTeamDamageSurvived", "killingSprees", "killsNearEnemyTurret",
    "killsOnLanersEarlyJungleAsJungler", "killsOnOtherLanesEarlyJungleAsLaner",
    "killsOnRecentlyHealedByAramPack", "killsUnderOwnTurret",
    "killsWithHelpFromEpicMonster", "knockEnemyIntoTeamAndKill",
    "landSkillShotsEarlyGame", "laneMinionsFirst10Minutes",
    "laningPhaseGoldExpAdvantage", "legendaryCount", "legendaryItemUsed",
    "lostAnInhibitor", "maxCsAdvantageOnLaneOpponent", "maxKillDeficit",
    "maxLevelLeadLaneOpponent", "mejaisFullStackInTime", "moreEnemyJungleThanOpponent",
    "multiKillOneSpell", "multiTurretRiftHeraldCount", "multikills",
    "multikillsAfterAggressiveFlash", "outerTurretExecutesBefore10Minutes",
    "outnumberedKills", "outnumberedNexusKill", "perfectDragonSoulsTaken",
    "perfectGame", "pickKillWithAlly", "playedChampSelectPosition", "poroExplosions",
    "quickCleanse", "quickFirstTurret", "quickSoloKills", "riftHeraldTakedowns",
    "saveAllyFromDeath", "scuttleCrabKills", "shortestTimeToAceFromFirstTakedown",
    "skillshotsDodged", "skillshotsHit", "snowballsHit", "soloBaronKills", "soloKills",
    "soloTurretsLategame", "stealthWardsPlaced", "survivedSingleDigitHpCount",
    "survivedThreeImmobilizesInFight", "takedownOnFirstTurret", "takedowns",
    "takedownsAfterGainingLevelAdvantage", "takedownsBeforeJungleMinionSpawn",
    "takedownsFirstXMinutes", "takedownsInAlcove", "takedownsInEnemyFountain",
    "teleportTakedowns", "teamBaronKills", "teamDamagePercentage",
    "teamElderDragonKills", "teamRiftHeraldKills", "thirdInhibitorDestroyedTime",
    "tookLargeDamageSurvived", "turretPlatesTaken", "turretTakedowns",
    "turretsTakenWithRiftHerald", "twentyMinionsIn3SecondsCount",
    "twoWardsOneSweeperCount", "unseenRecalls", "visionScoreAdvantageLaneOpponent",
    "visionScorePerMinute", "voidMonsterKill", "wardTakedowns",
    "wardTakedownsBefore20M", "wardsGuarded",
)

LEGENDARY_FIELD = "legendaryItemUsed"


def test_challenges_model_field_order_and_optionality() -> None:
    assert tuple(Challenges.model_fields) == EXPECTED_CHALLENGE_FIELDS
    # every challenge field is optional (defaults to None)
    assert [n for n, f in Challenges.model_fields.items() if f.is_required()] == []


def test_challenges_alias_only_on_12_assist_streak() -> None:
    aliases = {n: f.alias for n, f in Challenges.model_fields.items() if f.alias}
    assert aliases == {"x12AssistStreakCount": "12AssistStreakCount"}


def test_challenges_forbids_extra_and_accepts_alias_or_name() -> None:
    # extra="forbid": unmodelled keys must raise
    with pytest.raises(ValidationError):
        Challenges.model_validate({"someBrandNewChallenge": 1})
    # populate_by_name: both the wire alias and the field name validate
    assert Challenges.model_validate({"12AssistStreakCount": 4}).x12AssistStreakCount == 4
    assert Challenges.model_validate({"x12AssistStreakCount": 5}).x12AssistStreakCount == 5


def test_tabulated_challenges_column_order() -> None:
    expected = ("matchId", "teamId", "puuid", *EXPECTED_CHALLENGE_FIELDS)
    assert tuple(TabulatedParticipantChallenges.__annotations__) == expected
    assert columns_from_typed_dict(TabulatedParticipantChallenges) == expected


def test_challenge_numeric_fields_excludes_legendary() -> None:
    assert _CHALLENGE_NUMERIC_FIELDS == tuple(
        n for n in EXPECTED_CHALLENGE_FIELDS if n != LEGENDARY_FIELD
    )


def test_parser_coerces_numeric_floats_and_legendary_list() -> None:
    dump = {n: None for n in EXPECTED_CHALLENGE_FIELDS}
    dump["kda"] = 3  # int -> float
    dump["damagePerMinute"] = 2.5
    dump[LEGENDARY_FIELD] = [1, 2.0, "skip", 3]
    participant = SimpleNamespace(
        teamId=100,
        puuid="PUUID",
        challenges=SimpleNamespace(model_dump=lambda: dump),
    )

    rows = ParticipantChallengesParser().parse([participant], "EUW1_1")
    assert len(rows) == 1
    row = rows[0]
    assert row["matchId"] == "EUW1_1"
    assert row["teamId"] == 100
    assert row["puuid"] == "PUUID"
    assert row["kda"] == 3.0 and isinstance(row["kda"], float)
    assert row["damagePerMinute"] == 2.5
    assert row["kills"] if False else row["acesBefore15Minutes"] is None
    # legendaryItemUsed coerced to list[int], non-numeric dropped
    assert row[LEGENDARY_FIELD] == [1, 2, 3]
