from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Sequence, TypedDict, cast

from pydantic import NonNegativeInt, PositiveInt, ValidationError

from app.services.riot_api_client.parsers.base_parsers import (
    InfoParser,
    ParticipantParser,
)
from app.services.riot_api_client.parsers.models.non_timeline import (
    ChallengeValue,
    Info,
    Metadata,
    NonTimeline,
    Participant,
)
from app.services.riot_api_client.parsers.schema_drift import (
    non_timeline,
)

logger = logging.getLogger(__name__)


class TabulatedMetadata(TypedDict):
    matchId: str
    dataVersion: str
    participants: list[str]


class MetadataParser:
    def parse(self, validated: Metadata) -> list[TabulatedMetadata]:
        return [
            {
                "matchId": validated.matchId,
                "dataVersion": validated.dataVersion,
                "participants": validated.participants,
            }
        ]


class TabulatedInfo(TypedDict):
    endOfGameResult: str
    gameCreation: NonNegativeInt
    gameDuration: NonNegativeInt
    gameEndTimestamp: NonNegativeInt
    matchId: NonNegativeInt
    gameStartTimestamp: NonNegativeInt
    gameType: str
    gameVersion: str
    season: str
    patch: str
    subVersion: str
    mapId: PositiveInt
    platformId: str
    queueId: PositiveInt


class GameInfoParser:
    def parse(self, validated: Info) -> list[TabulatedInfo]:
        gameVersion = validated.gameVersion
        parts = gameVersion.split(".")
        if len(parts) >= 3:
            season = parts[0]
            patch = parts[1]
            subVersion = ".".join(parts[2:])
        else:
            season = "unknown"
            patch = "unknown"
            subVersion = "unknown"
            logger.warning(
                "UnexpectedGameVersionFormat match_id=%s gameVersion=%r",
                validated.gameId,
                gameVersion,
            )

        return [
            {
                "endOfGameResult": validated.endOfGameResult,
                "gameCreation": validated.gameCreation,
                "gameDuration": validated.gameDuration,
                "gameEndTimestamp": validated.gameEndTimestamp,
                "matchId": validated.gameId,
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


class TabulatedBan(TypedDict):
    matchId: NonNegativeInt
    teamId: PositiveInt
    pickTurn: PositiveInt
    championId: int


class BansParser:
    def parse(self, validated: Info) -> list[TabulatedBan]:
        matchId: NonNegativeInt = validated.gameId
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


class TabulatedFeat(TypedDict):
    matchId: NonNegativeInt
    teamId: PositiveInt
    featType: str
    featState: int


class FeatsParser:
    def parse(self, validated: Info) -> list[TabulatedFeat]:
        matchId: NonNegativeInt = validated.gameId
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


class TabulatedObjective(TypedDict):
    matchId: NonNegativeInt
    teamId: PositiveInt
    objectiveType: str
    first: bool
    kills: NonNegativeInt


class ObjectivesParser:
    def parse(self, validated: Info) -> list[TabulatedObjective]:
        rows: list[TabulatedObjective] = []
        matchId: NonNegativeInt = validated.gameId

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


class TabulatedParticipantStats(TypedDict):
    matchId: NonNegativeInt
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
    gameEndedInSurrender: bool
    teamEarlySurrendered: bool

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
        matchId: NonNegativeInt,
    ) -> list[TabulatedParticipantStats]:
        rows: list[TabulatedParticipantStats] = []

        complex = {"missions", "challenges", "perks"}
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
            "eligibleForProgressionindividualPosition",
            "lane",
            "role",
            "championName",
        }
        exclude_fields = complex | simple

        for p in participants:
            data = p.model_dump(
                exclude=exclude_fields,
            )
            data["matchId"] = matchId
            for field_name in self._UINT8_CLAMP_FIELDS:
                value = data.get(field_name)
                if value is not None and value > 255:
                    data[field_name] = 255

            rows.append(cast(TabulatedParticipantStats, data))

        return rows


class TabulatedParticipantChallenges(TypedDict):
    matchId: NonNegativeInt
    teamId: PositiveInt
    puuid: str
    payload: dict[str, ChallengeValue]


class ParticipantChallengesParser:
    def parse(
        self, participants: Sequence[Participant], matchId: NonNegativeInt
    ) -> list[TabulatedParticipantChallenges]:
        rows: list[TabulatedParticipantChallenges] = []
        for p in participants:
            teamId: PositiveInt = p.teamId
            puuid: str = p.puuid
            payload = {
                k: v
                for k, v in p.challenges.model_dump(
                    by_alias=True,
                    exclude_none=True,
                ).items()
                if not k.startswith("SWARM")
            }

            rows.append(
                {
                    "matchId": matchId,
                    "teamId": teamId,
                    "puuid": puuid,
                    "payload": payload,
                }
            )
        return rows


class TabulatedParticipantPerkValues(TypedDict):
    matchId: NonNegativeInt
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


class TabulatedParticipantPerkIds(TypedDict):
    matchId: NonNegativeInt
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
        self, participants: Sequence[Participant], matchId: NonNegativeInt
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
        self, participants: Sequence[Participant], matchId: NonNegativeInt
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

    def run(self, raw: dict[str, Any]) -> NonTimelineTables:
        metadata_raw = raw.get("metadata", {})
        match_id = (
            metadata_raw.get("matchId", "unknown")
            if isinstance(metadata_raw, dict)
            else "unknown"
        )
        drift_date = self._drift_date(raw)
        non_timeline(raw, match_id=match_id, drift_date=drift_date)

        try:
            nt = NonTimeline.model_validate(raw)
            metadata: Metadata = nt.metadata
            info: Info = nt.info
            participants: list[Participant] = info.participants
            matchId: NonNegativeInt = info.gameId

            tables = NonTimelineTables(
                metadata=self.metadata.parse(metadata),
                game_info=self.gameInfo.parse(info),
                bans=self.bans.parse(info),
                feats=self.feats.parse(info),
                objectives=self.objectives.parse(info),
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
            logger.warning(
                "SchemaValidation non_timeline match_id=%s date=%s errors=%s",
                match_id,
                drift_date,
                e.errors(include_input=False),
            )
            logger.warning(
                "Skipping non_timeline payload for match_id=%s due to validation errors "
                "during initial schema tuning.",
                match_id,
            )
            # TODO: Re-enable hard failure (raise ValueError) after initial schema
            # tuning is complete and drift has stabilized.
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
        return tables
