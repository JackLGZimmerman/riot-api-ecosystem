from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from pydantic import (
    NonNegativeInt,
    PositiveInt,
    ValidationError,
)

from app.services.riot_api_client.parsers.base_parsers import (
    EventParser,
    ParticipantParser,
)
from app.services.riot_api_client.parsers.models.non_timeline import (
    Feats,
    Info,
    Metadata,
    NonTimeline,
    Participant,
)


class TabulatedMetadata(TypedDict):
    matchId: str
    dataVersion: str
    participants: list[str]


class MetadataParser:
    def parse(self, validated: Metadata) -> TabulatedMetadata:
        return {
            "matchId": validated.matchId,
            "dataVersion": validated.dataVersion,
            "participants": validated.participants,
        }


class TabulatedInfo(TypedDict):
    endOfGameResult: str
    gameCreation: NonNegativeInt
    gameDuration: NonNegativeInt
    gameEndTimestamp: NonNegativeInt
    gameId: NonNegativeInt
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
    def parse(self, validated: Info) -> TabulatedInfo:
        gameVersion = validated.gameVersion
        season, patch, subVersion = gameVersion.split(".", 2)

        return {
            "endOfGameResult": validated.endOfGameResult,
            "gameCreation": validated.gameCreation,
            "gameDuration": validated.gameDuration,
            "gameEndTimestamp": validated.gameEndTimestamp,
            "gameId": validated.gameId,
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


class TabulatedBan(TypedDict):
    gameId: NonNegativeInt
    teamId: PositiveInt
    pickTurn: PositiveInt
    championId: int


class BansParser:
    def parse(self, validated: Info) -> list[TabulatedBan]:
        gameId: NonNegativeInt = validated.gameId
        rows: list[TabulatedBan] = []

        for team in validated.teams:
            teamId: PositiveInt = team.teamId
            for ban in team.bans:
                rows.append(
                    {
                        "gameId": gameId,
                        "teamId": teamId,
                        "pickTurn": ban.pickTurn,
                        "championId": ban.championId,
                    }
                )

        return rows


class TabulatedFeat(TypedDict):
    gameId: NonNegativeInt
    teamId: PositiveInt
    featType: str
    featState: int


class FeatsParser:
    def parse(self, validated: Info) -> list[TabulatedFeat]:
        gameId: NonNegativeInt = validated.gameId
        rows: list[TabulatedFeat] = []

        for team in validated.teams:
            teamId: PositiveInt = team.teamId
            feats: Feats = team.feats

            dumped: dict[str, Any] = feats.model_dump()
            for feat_type, feat_state in dumped.items():
                rows.append(
                    cast(
                        TabulatedFeat,
                        {
                            "gameId": gameId,
                            "teamId": teamId,
                            "featType": feat_type,
                            "featState": feat_state["featState"],
                        },
                    )
                )

        return rows


class TabulatedObjective(TypedDict):
    gameId: NonNegativeInt
    teamId: PositiveInt
    objectiveType: str
    first: bool
    kills: NonNegativeInt


class ObjectivesParser:
    def parse(self, validated: Info) -> list[TabulatedObjective]:
        rows: list[TabulatedObjective] = []
        gameId: NonNegativeInt = validated.gameId

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
                    "gameId": gameId,
                    "teamId": teamId,
                    "objectiveType": objectiveType,
                    "first": obj.first,
                    "kills": obj.kills,
                }
                rows.append(cast(TabulatedObjective, row))

        return rows


class TabulatedParticipantStats(TypedDict):
    gameId: NonNegativeInt
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
    damageDealtToEpicMonsters: NonNegativeInt

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

    roleBoundItem: NonNegativeInt

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
    retreatPings: NonNegativeInt

    unrealKills: NonNegativeInt


class ParticipantStatsParser:
    def parse(
        self,
        participants: list[Participant],
        gameId: NonNegativeInt,
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
            "summonerId",
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
            data["gameId"] = gameId

            rows.append(cast(TabulatedParticipantStats, data))

        return rows


class TabulatedParticipantChallenges(TypedDict):
    gameId: NonNegativeInt
    teamId: PositiveInt
    puuid: str


class ParticipantChallengesParser:
    def parse(
        self, participants: list[Participant], gameId: NonNegativeInt
    ) -> list[TabulatedParticipantChallenges]:
        rows: list[TabulatedParticipantChallenges] = []
        for p in participants:
            teamId: PositiveInt = p.teamId
            puuid: str = p.puuid

            data = p.challenges.model_dump()
            data["gameId"] = gameId
            data["teamId"] = teamId
            data["puuid"] = puuid

            rows.append(data)
        return rows


class TabulatedParticipantPerks(TypedDict):
    gameId: NonNegativeInt
    teamId: PositiveInt
    puuid: str

    stat_defense: int
    stat_flex: int
    stat_offense: int

    primary_style: int
    sub_style: int

    primary_perk_1: int
    primary_var1_1: int
    primary_var2_1: int
    primary_var3_1: int
    primary_perk_2: int
    primary_var1_2: int
    primary_var2_2: int
    primary_var3_2: int
    primary_perk_3: int
    primary_var1_3: int
    primary_var2_3: int
    primary_var3_3: int
    primary_perk_4: int
    primary_var1_4: int
    primary_var2_4: int
    primary_var3_4: int

    sub_perk_1: int
    sub_var1_1: int
    sub_var2_1: int
    sub_var3_1: int
    sub_perk_2: int
    sub_var1_2: int
    sub_var2_2: int
    sub_var3_2: int


class ParticipantPerksParser:
    def parse(
        self, participants: list[Participant], gameId: NonNegativeInt
    ) -> list[TabulatedParticipantPerks]:
        rows: list[TabulatedParticipantPerks] = []

        for p in participants:
            perks = p.perks
            stat = perks.statPerks

            style_by_desc = {s.description: s for s in perks.styles}
            primary = style_by_desc["primaryStyle"]
            sub = style_by_desc["subStyle"]

            payload: dict[str, Any] = {
                "gameId": gameId,
                "teamId": p.teamId,
                "puuid": p.puuid,
                "stat_defense": stat.defense,
                "stat_flex": stat.flex,
                "stat_offense": stat.offense,
                "primary_style": primary.style,
                "sub_style": sub.style,
            }

            def add_selections(prefix: str, selections: list[Any], count: int) -> None:
                for i in range(count):
                    sel = selections[i]
                    n = i + 1
                    payload[f"{prefix}_perk_{n}"] = sel.perk
                    payload[f"{prefix}_var1_{n}"] = sel.var1
                    payload[f"{prefix}_var2_{n}"] = sel.var2
                    payload[f"{prefix}_var3_{n}"] = sel.var3

            add_selections("primary", primary.selections, 4)
            add_selections("sub", sub.selections, 2)

            rows.append(cast(TabulatedParticipantPerks, payload))

        return rows


@dataclass
class NonTimelineTables:
    metadata: TabulatedMetadata
    game_info: TabulatedInfo
    bans: list[TabulatedBan]
    feats: list[TabulatedFeat]
    objectives: list[TabulatedObjective]
    participant_stats: list[TabulatedParticipantStats]
    participant_challenges: list[TabulatedParticipantChallenges]
    participant_perks: list[TabulatedParticipantPerks]


@dataclass(frozen=True)
class MatchDataNonTimelineParsingOrchestrator:
    metadata: EventParser[Metadata, TabulatedMetadata]
    gameInfo: EventParser[Info, TabulatedInfo]
    bans: EventParser[Info, list[TabulatedBan]]
    feats: EventParser[Info, list[TabulatedFeat]]
    objectives: EventParser[Info, list[TabulatedObjective]]
    participantStats: ParticipantParser[list[TabulatedParticipantStats]]
    participantChallenges: ParticipantParser[list[TabulatedParticipantChallenges]]
    participantPerks: ParticipantParser[list[TabulatedParticipantPerks]]

    def run(self, raw: dict[str, Any]) -> NonTimelineTables:
        try:
            nt = NonTimeline.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"raw did not match NonTimeline schema: {e}") from e

        metadata: Metadata = nt.metadata
        info: Info = nt.info
        participants: list[Participant] = info.participants
        gameId: NonNegativeInt = info.gameId

        return NonTimelineTables(
            metadata=self.metadata.parse(metadata),
            game_info=self.gameInfo.parse(info),
            bans=self.bans.parse(info),
            feats=self.feats.parse(info),
            objectives=self.objectives.parse(info),
            participant_stats=self.participantStats.parse(participants, gameId),
            participant_challenges=self.participantChallenges.parse(
                participants, gameId
            ),
            participant_perks=self.participantPerks.parse(participants, gameId),
        )


if __name__ == "__main__":
    path = Path("non-timeline.example.json")

    def load_dummy_non_timeline():
        with path.open("r") as f:
            data = json.load(f)

        return data

    data = load_dummy_non_timeline()
    validated_data = NonTimeline.model_validate(data)
