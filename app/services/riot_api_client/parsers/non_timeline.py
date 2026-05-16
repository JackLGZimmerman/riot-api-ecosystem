from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from pydantic import NonNegativeInt, PositiveInt, ValidationError

from app.services.riot_api_client.parsers.base_parsers import (
    InfoParser,
    ParticipantParser,
)
from app.services.riot_api_client.parsers.models.non_timeline import (
    Info,
    Metadata,
    NonTimeline,
    Participant,
)
from app.services.riot_api_client.parsers.schema_drift import (
    non_timeline,
)

logger = logging.getLogger(__name__)


def is_abort_payload(raw: dict[str, Any]) -> bool:
    info = raw.get("info")
    end_of_game_result = info.get("endOfGameResult") if isinstance(info, dict) else None
    return isinstance(end_of_game_result, str) and end_of_game_result.startswith("Abort")


class MatchIdRow(TypedDict, total=False):
    matchId: str | NonNegativeInt


class TabulatedMetadata(MatchIdRow):
    dataVersion: NonNegativeInt
    participants: list[str]


class MetadataParser:
    def parse(
        self,
        validated: Metadata,
        matchId: str | int,
    ) -> list[TabulatedMetadata]:
        _ = matchId
        return [
            {
                "matchId": validated.matchId,
                "dataVersion": int(validated.dataVersion),
                "participants": validated.participants,
            }
        ]


class TabulatedInfo(MatchIdRow):
    endOfGameResult: str
    gameCreation: NonNegativeInt
    gameDuration: NonNegativeInt
    gameEndTimestamp: NonNegativeInt
    gameId: NonNegativeInt
    gameStartTimestamp: NonNegativeInt
    gameType: str
    gameVersion: str
    season: int
    patch: int
    subVersion: str
    mapId: PositiveInt
    platformId: str
    queueId: PositiveInt


class GameInfoParser:
    def parse(
        self,
        validated: Info,
        matchId: str | int,
    ) -> list[TabulatedInfo]:
        gameVersion = validated.gameVersion
        parts = gameVersion.split(".")
        season = -1
        patch = -1
        subVersion = "unknown"
        if len(parts) >= 3:
            subVersion = ".".join(parts[2:])
            try:
                season = int(parts[0])
                patch = int(parts[1])
            except ValueError:
                logger.warning(
                    "UnexpectedGameVersionNumericFormat game_id=%s gameVersion=%r",
                    validated.gameId,
                    gameVersion,
                )
        else:
            logger.warning(
                "UnexpectedGameVersionFormat game_id=%s gameVersion=%r",
                validated.gameId,
                gameVersion,
            )

        return [
            {
                "endOfGameResult": validated.endOfGameResult,
                "gameCreation": validated.gameCreation,
                "gameDuration": validated.gameDuration,
                "gameEndTimestamp": validated.gameEndTimestamp,
                "gameId": validated.gameId,
                "matchId": matchId,
                "gameStartTimestamp": validated.gameStartTimestamp,
                "gameType": validated.gameType,
                "gameVersion": gameVersion,
                "season": season,
                "patch": patch,
                "subVersion": subVersion,
                "mapId": validated.mapId,
                "platformId": validated.platformId,
                "queueId": validated.queueId,
            }
        ]


class TabulatedBan(MatchIdRow):
    teamId: PositiveInt
    pickTurn: PositiveInt
    championId: int


class BansParser:
    def parse(
        self,
        validated: Info,
        matchId: str | int,
    ) -> list[TabulatedBan]:
        rows: list[TabulatedBan] = []

        for team in validated.teams:
            teamId: PositiveInt = team.teamId
            for ban in team.bans:
                rows.append(
                    {
                        "matchId": matchId,
                        "teamId": teamId,
                        "pickTurn": ban.pickTurn,
                        "championId": ban.championId,
                    }
                )

        return rows


class TabulatedFeat(MatchIdRow):
    teamId: PositiveInt
    featType: str
    featState: int


class FeatsParser:
    def parse(
        self,
        validated: Info,
        matchId: str | int,
    ) -> list[TabulatedFeat]:
        rows: list[TabulatedFeat] = []

        for team in validated.teams:
            teamId: PositiveInt = team.teamId
            feats = team.feats
            if feats is None:
                continue

            dumped: dict[str, Any] = feats.model_dump()
            for feat_type, feat_state in dumped.items():
                rows.append(
                    cast(
                        TabulatedFeat,
                        {
                            "matchId": matchId,
                            "teamId": teamId,
                            "featType": feat_type,
                            "featState": feat_state["featState"],
                        },
                    )
                )

        return rows


class TabulatedObjective(MatchIdRow):
    teamId: PositiveInt
    objectiveType: str
    first: bool
    kills: NonNegativeInt


class ObjectivesParser:
    def parse(
        self,
        validated: Info,
        matchId: str | int,
    ) -> list[TabulatedObjective]:
        rows: list[TabulatedObjective] = []

        for team in validated.teams:
            teamId: PositiveInt = team.teamId
            objectives = team.objectives
            if objectives is None:
                continue

            for objectiveType in (
                "baron",
                "champion",
                "dragon",
                "horde",
                "inhibitor",
                "riftHerald",
                "tower",
            ):
                obj = getattr(objectives, objectiveType, None)
                if obj is None:
                    continue

                row = {
                    "matchId": matchId,
                    "teamId": teamId,
                    "objectiveType": objectiveType,
                    "first": obj.first,
                    "kills": obj.kills,
                }
                rows.append(cast(TabulatedObjective, row))

        return rows


class TabulatedParticipantStats(MatchIdRow):
    participantId: NonNegativeInt
    puuid: str
    teamId: NonNegativeInt

    summonerId: str
    summonerLevel: NonNegativeInt
    summonerName: str

    riotIdGameName: str
    riotIdTagline: str

    profileIcon: NonNegativeInt

    championId: NonNegativeInt
    championTransform: NonNegativeInt

    champLevel: NonNegativeInt
    champExperience: NonNegativeInt

    teamPosition: str

    win: bool
    gameEndedInEarlySurrender: bool
    gameEndedInIGNBSurrender: bool | None
    gameEndedInSurrender: bool
    causedGameEndFromIGNBSurrender: bool | None
    teamEarlySurrendered: bool
    teamIGNBSurrendered: bool | None
    wasPremadeWithIGNBGameEndCauser: bool | None
    wasPremadeWithSevereTransgressor: bool | None
    wasSevereTransgressor: bool | None

    kills: NonNegativeInt
    deaths: NonNegativeInt
    assists: NonNegativeInt

    doubleKills: NonNegativeInt
    tripleKills: NonNegativeInt
    quadraKills: NonNegativeInt
    pentaKills: NonNegativeInt

    killingSprees: NonNegativeInt
    largestKillingSpree: NonNegativeInt
    largestMultiKill: NonNegativeInt
    largestCriticalStrike: NonNegativeInt

    firstBloodKill: bool
    firstBloodAssist: bool
    firstTowerKill: bool
    firstTowerAssist: bool

    goldEarned: NonNegativeInt
    goldSpent: NonNegativeInt
    consumablesPurchased: NonNegativeInt
    itemsPurchased: NonNegativeInt

    item0: NonNegativeInt
    item1: NonNegativeInt
    item2: NonNegativeInt
    item3: NonNegativeInt
    item4: NonNegativeInt
    item5: NonNegativeInt
    item6: NonNegativeInt

    totalDamageDealt: NonNegativeInt
    totalDamageDealtToChampions: NonNegativeInt
    physicalDamageDealt: NonNegativeInt
    physicalDamageDealtToChampions: NonNegativeInt
    magicDamageDealt: NonNegativeInt
    magicDamageDealtToChampions: NonNegativeInt
    trueDamageDealt: NonNegativeInt
    trueDamageDealtToChampions: NonNegativeInt

    damageDealtToBuildings: NonNegativeInt
    damageDealtToTurrets: NonNegativeInt
    damageDealtToObjectives: NonNegativeInt
    damageDealtToEpicMonsters: int | None

    totalDamageTaken: NonNegativeInt
    physicalDamageTaken: NonNegativeInt
    magicDamageTaken: NonNegativeInt
    trueDamageTaken: NonNegativeInt

    damageSelfMitigated: NonNegativeInt

    totalHeal: NonNegativeInt
    totalHealsOnTeammates: NonNegativeInt
    totalUnitsHealed: NonNegativeInt

    totalDamageShieldedOnTeammates: NonNegativeInt

    timeCCingOthers: NonNegativeInt
    totalTimeCCDealt: NonNegativeInt

    totalMinionsKilled: NonNegativeInt
    neutralMinionsKilled: NonNegativeInt
    totalAllyJungleMinionsKilled: NonNegativeInt
    totalEnemyJungleMinionsKilled: NonNegativeInt

    baronKills: NonNegativeInt
    dragonKills: NonNegativeInt

    inhibitorKills: NonNegativeInt
    inhibitorTakedowns: NonNegativeInt
    inhibitorsLost: NonNegativeInt

    turretKills: NonNegativeInt
    turretTakedowns: NonNegativeInt
    turretsLost: NonNegativeInt

    objectivesStolen: NonNegativeInt
    objectivesStolenAssists: NonNegativeInt

    visionScore: NonNegativeInt
    wardsPlaced: NonNegativeInt
    wardsKilled: NonNegativeInt
    detectorWardsPlaced: NonNegativeInt
    sightWardsBoughtInGame: NonNegativeInt
    visionWardsBoughtInGame: NonNegativeInt
    visionClearedPings: NonNegativeInt

    summoner1Id: NonNegativeInt
    summoner2Id: NonNegativeInt
    summoner1Casts: NonNegativeInt
    summoner2Casts: NonNegativeInt

    spell1Casts: NonNegativeInt
    spell2Casts: NonNegativeInt
    spell3Casts: NonNegativeInt
    spell4Casts: NonNegativeInt

    roleBoundItem: int | None
    bountyLevel: int | None
    playerBehaviorIsHeroInCombat: NonNegativeInt | None

    timePlayed: NonNegativeInt
    totalTimeSpentDead: NonNegativeInt
    longestTimeSpentLiving: NonNegativeInt

    allInPings: NonNegativeInt
    assistMePings: NonNegativeInt
    basicPings: NonNegativeInt
    commandPings: NonNegativeInt
    dangerPings: NonNegativeInt
    enemyMissingPings: NonNegativeInt
    enemyVisionPings: NonNegativeInt
    getBackPings: NonNegativeInt
    holdPings: NonNegativeInt
    needVisionPings: NonNegativeInt
    onMyWayPings: NonNegativeInt
    pushPings: NonNegativeInt
    retreatPings: NonNegativeInt | None

    unrealKills: NonNegativeInt


class ParticipantStatsParser:
    _UINT8_CLAMP_FIELDS = {
        "visionScore",
        "wardsPlaced",
        "wardsKilled",
        "allInPings",
        "assistMePings",
        "basicPings",
        "commandPings",
        "dangerPings",
        "enemyMissingPings",
        "enemyVisionPings",
        "getBackPings",
        "holdPings",
        "needVisionPings",
        "onMyWayPings",
        "pushPings",
        "retreatPings",
        "unrealKills",
    }

    def parse(
        self,
        participants: Sequence[Participant],
        matchId: str | NonNegativeInt,
    ) -> list[TabulatedParticipantStats]:
        rows: list[TabulatedParticipantStats] = []

        complex = {"missions", "challenges", "perks", "PlayerBehavior"}
        simple = {
            "PlayerScore0",
            "PlayerScore1",
            "PlayerScore2",
            "PlayerScore3",
            "PlayerScore4",
            "PlayerScore5",
            "PlayerScore6",
            "PlayerScore7",
            "PlayerScore8",
            "PlayerScore9",
            "PlayerScore10",
            "PlayerScore11",
            "placement",
            "playerAugment1",
            "playerAugment2",
            "playerAugment3",
            "playerAugment4",
            "playerAugment5",
            "playerAugment6",
            "playerSubteamId",
            "subteamPlacement",
            "nexusKills",
            "nexusTakedowns",
            "nexusLost",
            "eligibleForProgression",
            "individualPosition",
            "lane",
            "role",
            "championName",
        }
        exclude_fields = complex | simple

        for p in participants:
            data = p.model_dump(
                exclude=exclude_fields,
            )
            player_behavior = p.PlayerBehavior
            data["playerBehaviorIsHeroInCombat"] = (
                player_behavior.PlayerBehavior_IsHeroInCombat
                if player_behavior is not None
                else None
            )
            data["matchId"] = matchId
            for field_name in self._UINT8_CLAMP_FIELDS:
                value = data.get(field_name)
                if value is not None and value > 255:
                    data[field_name] = 255

            rows.append(cast(TabulatedParticipantStats, data))

        return rows


class TabulatedParticipantChallenges(MatchIdRow):
    teamId: PositiveInt
    puuid: str

    x12AssistStreakCount: float | None
    HealFromMapSources: float | None
    InfernalScalePickup: float | None
    abilityUses: float | None
    acesBefore15Minutes: float | None
    alliedJungleMonsterKills: float | None
    baronBuffGoldAdvantageOverThreshold: float | None
    baronTakedowns: float | None
    blastConeOppositeOpponentCount: float | None
    bountyGold: float | None
    buffsStolen: float | None
    completeSupportQuestInTime: float | None
    controlWardTimeCoverageInRiverOrEnemyHalf: float | None
    controlWardsPlaced: float | None
    damagePerMinute: float | None
    damageTakenOnTeamPercentage: float | None
    dancedWithRiftHerald: float | None
    deathsByEnemyChamps: float | None
    dodgeSkillShotsSmallWindow: float | None
    doubleAces: float | None
    dragonTakedowns: float | None
    earliestBaron: float | None
    earliestDragonTakedown: float | None
    earliestElderDragon: float | None
    earlyLaningPhaseGoldExpAdvantage: float | None
    effectiveHealAndShielding: float | None
    elderDragonKillsWithOpposingSoul: float | None
    elderDragonMultikills: float | None
    enemyChampionImmobilizations: float | None
    enemyJungleMonsterKills: float | None
    epicMonsterKillsNearEnemyJungler: float | None
    epicMonsterKillsWithin30SecondsOfSpawn: float | None
    epicMonsterSteals: float | None
    epicMonsterStolenWithoutSmite: float | None
    firstTurretKilled: float | None
    firstTurretKilledTime: float | None
    fasterSupportQuestCompletion: float | None
    fastestLegendary: float | None
    fistBumpParticipation: float | None
    flawlessAces: float | None
    fullTeamTakedown: float | None
    gameLength: float | None
    getTakedownsInAllLanesEarlyJungleAsLaner: float | None
    goldPerMinute: float | None
    hadOpenNexus: float | None
    hadAfkTeammate: float | None
    highestChampionDamage: float | None
    highestCrowdControlScore: float | None
    highestWardKills: float | None
    immobilizeAndKillWithAlly: float | None
    initialBuffCount: float | None
    initialCrabCount: float | None
    jungleCsBefore10Minutes: float | None
    junglerKillsEarlyJungle: float | None
    junglerTakedownsNearDamagedEpicMonster: float | None
    kTurretsDestroyedBeforePlatesFall: float | None
    kda: float | None
    killAfterHiddenWithAlly: float | None
    killParticipation: float | None
    killedChampTookFullTeamDamageSurvived: float | None
    killingSprees: float | None
    killsNearEnemyTurret: float | None
    killsOnLanersEarlyJungleAsJungler: float | None
    killsOnOtherLanesEarlyJungleAsLaner: float | None
    killsOnRecentlyHealedByAramPack: float | None
    killsUnderOwnTurret: float | None
    killsWithHelpFromEpicMonster: float | None
    knockEnemyIntoTeamAndKill: float | None
    landSkillShotsEarlyGame: float | None
    laneMinionsFirst10Minutes: float | None
    laningPhaseGoldExpAdvantage: float | None
    legendaryCount: float | None
    legendaryItemUsed: list[int]
    lostAnInhibitor: float | None
    maxCsAdvantageOnLaneOpponent: float | None
    maxKillDeficit: float | None
    maxLevelLeadLaneOpponent: float | None
    mejaisFullStackInTime: float | None
    moreEnemyJungleThanOpponent: float | None
    multiKillOneSpell: float | None
    multiTurretRiftHeraldCount: float | None
    multikills: float | None
    multikillsAfterAggressiveFlash: float | None
    outerTurretExecutesBefore10Minutes: float | None
    outnumberedKills: float | None
    outnumberedNexusKill: float | None
    perfectDragonSoulsTaken: float | None
    perfectGame: float | None
    pickKillWithAlly: float | None
    playedChampSelectPosition: float | None
    poroExplosions: float | None
    quickCleanse: float | None
    quickFirstTurret: float | None
    quickSoloKills: float | None
    riftHeraldTakedowns: float | None
    saveAllyFromDeath: float | None
    scuttleCrabKills: float | None
    shortestTimeToAceFromFirstTakedown: float | None
    skillshotsDodged: float | None
    skillshotsHit: float | None
    snowballsHit: float | None
    soloBaronKills: float | None
    soloKills: float | None
    soloTurretsLategame: float | None
    stealthWardsPlaced: float | None
    survivedSingleDigitHpCount: float | None
    survivedThreeImmobilizesInFight: float | None
    takedownOnFirstTurret: float | None
    takedowns: float | None
    takedownsAfterGainingLevelAdvantage: float | None
    takedownsBeforeJungleMinionSpawn: float | None
    takedownsFirstXMinutes: float | None
    takedownsInAlcove: float | None
    takedownsInEnemyFountain: float | None
    teleportTakedowns: float | None
    teamBaronKills: float | None
    teamDamagePercentage: float | None
    teamElderDragonKills: float | None
    teamRiftHeraldKills: float | None
    thirdInhibitorDestroyedTime: float | None
    tookLargeDamageSurvived: float | None
    turretPlatesTaken: float | None
    turretTakedowns: float | None
    turretsTakenWithRiftHerald: float | None
    twentyMinionsIn3SecondsCount: float | None
    twoWardsOneSweeperCount: float | None
    unseenRecalls: float | None
    visionScoreAdvantageLaneOpponent: float | None
    visionScorePerMinute: float | None
    voidMonsterKill: float | None
    wardTakedowns: float | None
    wardTakedownsBefore20M: float | None
    wardsGuarded: float | None


_CHALLENGE_NUMERIC_FIELDS: tuple[str, ...] = tuple(
    k
    for k in TabulatedParticipantChallenges.__annotations__
    if k not in {"matchId", "teamId", "puuid", "legendaryItemUsed"}
)


class ParticipantChallengesParser:
    def parse(
        self, participants: Sequence[Participant], matchId: str | int
    ) -> list[TabulatedParticipantChallenges]:
        rows: list[TabulatedParticipantChallenges] = []
        for p in participants:
            dump = p.challenges.model_dump()
            row: dict[str, Any] = {
                "matchId": matchId,
                "teamId": p.teamId,
                "puuid": p.puuid,
            }
            for field_name in _CHALLENGE_NUMERIC_FIELDS:
                value = dump.get(field_name)
                row[field_name] = (
                    float(value) if isinstance(value, (int, float)) else None
                )
            legendary = dump.get("legendaryItemUsed")
            row["legendaryItemUsed"] = (
                [int(v) for v in legendary if isinstance(v, (int, float))]
                if isinstance(legendary, list)
                else []
            )
            rows.append(cast(TabulatedParticipantChallenges, row))
        return rows


class TabulatedParticipantPerkValues(MatchIdRow):
    teamId: PositiveInt
    puuid: str

    primary_var1_1: int
    primary_var2_1: int
    primary_var3_1: int
    primary_var1_2: int
    primary_var2_2: int
    primary_var3_2: int
    primary_var1_3: int
    primary_var2_3: int
    primary_var3_3: int
    primary_var1_4: int
    primary_var2_4: int
    primary_var3_4: int

    sub_var1_1: int
    sub_var2_1: int
    sub_var3_1: int
    sub_var1_2: int
    sub_var2_2: int
    sub_var3_2: int


class TabulatedParticipantPerkIds(MatchIdRow):
    teamId: PositiveInt
    puuid: str

    stat_defense: int
    stat_flex: int
    stat_offense: int

    primary_style: int
    sub_style: int

    primary_perk_1: int
    primary_perk_2: int
    primary_perk_3: int
    primary_perk_4: int

    sub_perk_1: int
    sub_perk_2: int
    perk_combo_key: int


class ParticipantPerkValuesParser:
    def parse(
        self, participants: Sequence[Participant], matchId: str | int
    ) -> list[TabulatedParticipantPerkValues]:
        rows: list[TabulatedParticipantPerkValues] = []

        for p in participants:
            base_payload = {
                "matchId": matchId,
                "teamId": p.teamId,
                "puuid": p.puuid,
            }
            style_by_desc = {s.description: s for s in p.perks.styles}
            primary = style_by_desc["primaryStyle"]
            sub = style_by_desc["subStyle"]

            primary_payload = {
                f"primary_var{var_idx}_{sel_idx}": getattr(sel, f"var{var_idx}")
                for sel_idx, sel in enumerate(primary.selections, start=1)
                for var_idx in (1, 2, 3)
            }
            sub_payload = {
                f"sub_var{var_idx}_{sel_idx}": getattr(sel, f"var{var_idx}")
                for sel_idx, sel in enumerate(sub.selections, start=1)
                for var_idx in (1, 2, 3)
            }

            rows.append(
                cast(
                    TabulatedParticipantPerkValues,
                    {
                        **base_payload,
                        **primary_payload,
                        **sub_payload,
                    },
                )
            )

        return rows


class ParticipantPerkIdsParser:
    def parse(
        self, participants: Sequence[Participant], matchId: str | int
    ) -> list[TabulatedParticipantPerkIds]:
        rows: list[TabulatedParticipantPerkIds] = []
        perk_bit_width = 14

        for p in participants:
            base_payload = {
                "matchId": matchId,
                "teamId": p.teamId,
                "puuid": p.puuid,
            }
            stat = p.perks.statPerks
            style_by_desc = {s.description: s for s in p.perks.styles}
            primary = style_by_desc["primaryStyle"]
            sub = style_by_desc["subStyle"]
            primary_perk_payload = {
                f"primary_perk_{idx}": sel.perk
                for idx, sel in enumerate(primary.selections, start=1)
            }
            sub_perk_payload = {
                f"sub_perk_{idx}": sel.perk
                for idx, sel in enumerate(sub.selections, start=1)
            }
            selected_perks = [
                *(sel.perk for sel in primary.selections),
                *(sel.perk for sel in sub.selections),
            ]
            combo_key = sum(
                int(perk_id) << (perk_bit_width * idx)
                for idx, perk_id in enumerate(selected_perks)
            )

            rows.append(
                cast(
                    TabulatedParticipantPerkIds,
                    {
                        **base_payload,
                        "stat_defense": stat.defense,
                        "stat_flex": stat.flex,
                        "stat_offense": stat.offense,
                        "primary_style": primary.style,
                        "sub_style": sub.style,
                        **primary_perk_payload,
                        **sub_perk_payload,
                        "perk_combo_key": combo_key,
                    },
                )
            )

        return rows


@dataclass
class NonTimelineTables:
    metadata: list[TabulatedMetadata]
    game_info: list[TabulatedInfo]
    bans: list[TabulatedBan]
    feats: list[TabulatedFeat]
    objectives: list[TabulatedObjective]
    participant_stats: list[TabulatedParticipantStats]
    participant_challenges: list[TabulatedParticipantChallenges]
    participant_perk_values: list[TabulatedParticipantPerkValues]
    participant_perk_ids: list[TabulatedParticipantPerkIds]


@dataclass(frozen=True)
class MatchDataNonTimelineParsingOrchestrator:
    metadata: InfoParser[Metadata, list[TabulatedMetadata]] = field(
        default_factory=MetadataParser
    )
    gameInfo: InfoParser[Info, list[TabulatedInfo]] = field(
        default_factory=GameInfoParser
    )
    bans: InfoParser[Info, list[TabulatedBan]] = field(default_factory=BansParser)
    feats: InfoParser[Info, list[TabulatedFeat]] = field(default_factory=FeatsParser)
    objectives: InfoParser[Info, list[TabulatedObjective]] = field(
        default_factory=ObjectivesParser
    )
    participantStats: ParticipantParser[list[TabulatedParticipantStats]] = field(
        default_factory=ParticipantStatsParser
    )
    participantChallenges: ParticipantParser[list[TabulatedParticipantChallenges]] = (
        field(default_factory=ParticipantChallengesParser)
    )
    participantPerkValues: ParticipantParser[list[TabulatedParticipantPerkValues]] = (
        field(default_factory=ParticipantPerkValuesParser)
    )
    participantPerkIds: ParticipantParser[list[TabulatedParticipantPerkIds]] = field(
        default_factory=ParticipantPerkIdsParser
    )

    @staticmethod
    def _drift_date(raw: dict[str, Any]) -> str:
        try:
            game_creation = raw["info"]["gameCreation"]
        except (KeyError, IndexError, TypeError):
            game_creation = 0

        if isinstance(game_creation, int) and game_creation > 0:
            return datetime.fromtimestamp(game_creation / 1000, tz=UTC).date().isoformat()

        return datetime.now(tz=UTC).date().isoformat()

    @staticmethod
    def _is_unsupported_game_mode(raw: dict[str, Any]) -> bool:
        # SWARM (gameMode=STRAWBERRY) ships a different challenges schema and is
        # never collected — short-circuit before drift detection so it does not
        # surface as drift noise.
        info = raw.get("info")
        game_mode = info.get("gameMode") if isinstance(info, dict) else None
        return game_mode == "STRAWBERRY"

    @staticmethod
    def _strip_excluded_challenge_keys(raw: dict[str, Any]) -> None:
        # Some ranked matches include SWARM_* challenge keys (value 0) because Riot
        # attaches mode-specific challenges globally. We don't want them — strip
        # before drift check and model validation so they never surface as errors.
        info = raw.get("info")
        if not isinstance(info, dict):
            return
        participants = info.get("participants")
        if not isinstance(participants, list):
            return
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            challenges = participant.get("challenges")
            if not isinstance(challenges, dict):
                continue
            for key in [k for k in challenges if k.startswith("SWARM_")]:
                del challenges[key]

    def run(self, raw: dict[str, Any]) -> NonTimelineTables:
        metadata_raw = raw.get("metadata", {})
        match_id = (
            metadata_raw.get("matchId", "unknown")
            if isinstance(metadata_raw, dict)
            else "unknown"
        )

        self._strip_excluded_challenge_keys(raw)

        if self._is_unsupported_game_mode(raw):
            logger.info(
                "NonTimelineSkip match_id=%s reason=unsupported_game_mode mode=STRAWBERRY",
                match_id,
            )
            return NonTimelineTables(
                metadata=[],
                game_info=[],
                bans=[],
                feats=[],
                objectives=[],
                participant_stats=[],
                participant_challenges=[],
                participant_perk_values=[],
                participant_perk_ids=[],
            )

        drift_date = self._drift_date(raw)
        non_timeline(raw, match_id=match_id, drift_date=drift_date)

        if is_abort_payload(raw):
            logger.warning(
                "NonTimelineAbort match_id=%s date=%s; skipping non-timeline rows.",
                match_id,
                drift_date,
            )
            return NonTimelineTables(
                metadata=[],
                game_info=[],
                bans=[],
                feats=[],
                objectives=[],
                participant_stats=[],
                participant_challenges=[],
                participant_perk_values=[],
                participant_perk_ids=[],
            )

        try:
            nt = NonTimeline.model_validate(raw)
            metadata: Metadata = nt.metadata
            info: Info = nt.info
            participants: list[Participant] = info.participants
            matchId = metadata.matchId

            tables = NonTimelineTables(
                metadata=self.metadata.parse(metadata, matchId),
                game_info=self.gameInfo.parse(info, matchId),
                bans=self.bans.parse(info, matchId),
                feats=self.feats.parse(info, matchId),
                objectives=self.objectives.parse(info, matchId),
                participant_stats=self.participantStats.parse(participants, matchId),
                participant_challenges=self.participantChallenges.parse(
                    participants, matchId
                ),
                participant_perk_values=self.participantPerkValues.parse(
                    participants, matchId
                ),
                participant_perk_ids=self.participantPerkIds.parse(participants, matchId),
            )
        except ValidationError as e:
            errs = e.errors(include_input=True)
            logger.warning(
                "SchemaValidation non_timeline match_id=%s date=%s errors=%s",
                match_id,
                drift_date,
                e.errors(include_input=False),
            )
            logger.warning(
                "SchemaValidation non_timeline value=%r",
                errs[-1].get("input") if errs else None,
            )
            logger.warning(
                "Aborting non_timeline payload for match_id=%s due to validation errors.",
                match_id,
            )
            raise ValueError(
                f"Schema validation failed for non_timeline payload match_id={match_id}"
            ) from e
        return tables
