from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, NonNegativeInt, RootModel, ValidationError

ChallengeValue = (
    NonNegativeInt | float | list[NonNegativeInt] | list[float] | str | bool | None
)

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


class EventParser(Protocol[InT, OutT]):
    def parse(self, validated: InT) -> OutT: ...


POutT = TypeVar("POutT", covariant=True)


class ParticipantParser(Protocol[POutT]):
    def parse(
        self, participants: list[Participant], gameId: NonNegativeInt
    ) -> POutT: ...


RawT = TypeVar("RawT", contravariant=True)
RunOutT = TypeVar("RunOutT", covariant=True)


class Orchestrator(Protocol[RawT, RunOutT]):
    def run(self, raw: RawT) -> RunOutT: ...


class TabulatedMetadata(BaseModel):
    matchId: str
    dataVersion: str
    participants: list[str]


class MetadataParser:
    def parse(self, validated: Metadata) -> TabulatedMetadata:
        return TabulatedMetadata(
            matchId=validated.matchId,
            dataVersion=validated.dataVersion,
            participants=validated.participants,
        )


class TabulatedInfo(BaseModel):
    endOfGameResult: EndOfGameResult
    gameCreation: NonNegativeInt
    gameDuration: NonNegativeInt
    gameEndTimestamp: NonNegativeInt
    gameId: NonNegativeInt
    gameStartTimestamp: NonNegativeInt
    gameType: GameType
    gameVersion: str
    season: str
    patch: str
    subVersion: str
    mapId: MapID
    platformId: PlatformID
    queueId: QueueID


class GameInfoParser:
    def parse(self, validated: Info) -> TabulatedInfo:
        gameVersion = validated.gameVersion
        season, patch, subVersion = gameVersion.split(".", 2)

        return TabulatedInfo(
            endOfGameResult=validated.endOfGameResult,
            gameCreation=validated.gameCreation,
            gameDuration=validated.gameDuration,
            gameEndTimestamp=validated.gameEndTimestamp,
            gameId=validated.gameId,
            gameStartTimestamp=validated.gameStartTimestamp,
            gameType=validated.gameType,
            gameVersion=validated.gameVersion,
            season=season,
            patch=patch,
            subVersion=subVersion,
            mapId=validated.mapId,
            platformId=validated.platformId,
            queueId=validated.queueId,
        )


class TabulatedBan(BaseModel):
    gameId: NonNegativeInt
    teamId: TeamID
    pickTurn: PickTurn
    championId: int


class BansParser:
    def parse(self, validated: Info) -> list[TabulatedBan]:
        teams = validated.teams
        gameId: NonNegativeInt = validated.gameId

        tabulated_bans: list[TabulatedBan] = []
        for team in teams:
            bans: list[Ban] = team.bans
            teamId: TeamID = team.teamId
            for ban in bans:
                tabulated_bans.append(
                    TabulatedBan(
                        gameId=gameId,
                        teamId=teamId,
                        pickTurn=ban.pickTurn,
                        championId=ban.championId,
                    )
                )
        return tabulated_bans


class TabulatedFeat(BaseModel):
    gameId: NonNegativeInt
    teamId: TeamID
    featType: str
    featState: int


class FeatsParser:
    def parse(self, validated: Info) -> list[TabulatedFeat]:
        tabulated_feats: list[TabulatedFeat] = []
        gameId: NonNegativeInt = validated.gameId

        for team in validated.teams:
            teamId: TeamID = team.teamId
            feats: Feats = team.feats

            for feat_type, feat_state in feats.model_dump().items():
                tabulated_feats.append(
                    TabulatedFeat(
                        gameId=gameId,
                        teamId=teamId,
                        featType=feat_type,
                        featState=feat_state["featState"],
                    )
                )

        return tabulated_feats


class TabulatedObjective(BaseModel):
    gameId: NonNegativeInt
    teamId: TeamID
    objectiveType: str
    first: bool
    kills: NonNegativeInt


class ObjectivesParser:
    def parse(self, validated: Info) -> list[TabulatedObjective]:
        rows: list[TabulatedObjective] = []
        gameId: NonNegativeInt = validated.gameId

        for team in validated.teams:
            teamId: TeamID = team.teamId
            objectives = team.objectives

            if not objectives:
                continue

            for obj_type, obj in objectives.model_dump().items():
                rows.append(
                    TabulatedObjective(
                        gameId=gameId,
                        teamId=teamId,
                        objectiveType=obj_type,
                        first=obj["first"],
                        kills=obj["kills"],
                    )
                )

        return rows


class TabulatedParticipantStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

            rows.append(TabulatedParticipantStats.model_validate(data))

        return rows


class TabulatedParticipantChallenges(BaseModel):
    model_config = ConfigDict(extra="allow")

    gameId: NonNegativeInt
    teamId: TeamID
    puuid: str


class ParticipantChallengesParser:
    def parse(
        self, participants: list[Participant], gameId: NonNegativeInt
    ) -> list[TabulatedParticipantChallenges]:
        rows: list[TabulatedParticipantChallenges] = []
        for p in participants:
            teamId: TeamID = p.teamId
            puuid: str = p.puuid

            data = p.challenges.model_dump()
            data["gameId"] = gameId
            data["teamId"] = teamId
            data["puuid"] = puuid

            rows.append(TabulatedParticipantChallenges.model_validate(data))
        return rows


class TabulatedParticipantPerks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gameId: NonNegativeInt
    teamId: TeamID
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
        self,
        participants: list[Participant],
        gameId: NonNegativeInt,
    ) -> list[TabulatedParticipantPerks]:
        rows: list[TabulatedParticipantPerks] = []

        for p in participants:
            perks = p.perks
            stat = perks.statPerks

            style_by_desc = {s.description: s for s in perks.styles}
            primary = style_by_desc["primaryStyle"]
            sub = style_by_desc["subStyle"]

            payload: dict[str, int | str] = {
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

            rows.append(TabulatedParticipantPerks.model_validate(payload))

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


class NonTimeline(BaseModel):
    metadata: Metadata
    info: Info


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataVersion: str
    matchId: str
    participants: list[str]


EndOfGameResult = Literal["GameComplete"]
GameMode = Literal["CLASSIC"]
GameType = Literal["MATCHED_GAME"]
MapID = Literal[11]
PlatformID = Literal[
    "BR1",
    "LA1",
    "LA2",
    "NA1",
    "EUW1",
    "EUN1",
    "RU",
    "TR1",
    "ME1",
    "JP1",
    "KR",
    "TW2",
    "OC1",
    "VN2",
    "SG2",
]
QueueID = Literal[420, 440]
PickTurn = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TeamID = Literal[100, 200]
IndividualPosition = Literal["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
Lane = Literal["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
Role = Literal["TOP", "NONE", "CARRY", "SOLO", "SUPPORT", "DUO"]
TeamPosition = Literal["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


class Challenges(RootModel[dict[str, ChallengeValue]]):
    pass


class Missions(BaseModel):
    playerScore0: NonNegativeInt
    playerScore1: NonNegativeInt
    playerScore2: NonNegativeInt
    playerScore3: NonNegativeInt
    playerScore4: NonNegativeInt
    playerScore5: NonNegativeInt
    playerScore6: NonNegativeInt
    playerScore7: NonNegativeInt
    playerScore8: NonNegativeInt
    playerScore9: NonNegativeInt
    playerScore10: NonNegativeInt
    playerScore11: NonNegativeInt


class StatPerks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defense: NonNegativeInt
    flex: NonNegativeInt
    offense: NonNegativeInt


class PerkSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    perk: NonNegativeInt
    var1: NonNegativeInt
    var2: NonNegativeInt
    var3: NonNegativeInt


class PerkStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    selections: list[PerkSelection]
    style: NonNegativeInt


class Perks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statPerks: StatPerks
    styles: list[PerkStyle]


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    PlayerScore0: NonNegativeInt
    PlayerScore1: NonNegativeInt
    PlayerScore10: NonNegativeInt
    PlayerScore11: NonNegativeInt
    PlayerScore2: NonNegativeInt
    PlayerScore3: NonNegativeInt
    PlayerScore4: NonNegativeInt
    PlayerScore5: NonNegativeInt
    PlayerScore6: NonNegativeInt
    PlayerScore7: NonNegativeInt
    PlayerScore8: NonNegativeInt
    PlayerScore9: NonNegativeInt
    allInPings: NonNegativeInt
    assistMePings: NonNegativeInt
    assists: NonNegativeInt
    baronKills: NonNegativeInt
    basicPings: NonNegativeInt
    challenges: Challenges
    champExperience: NonNegativeInt
    champLevel: NonNegativeInt
    championId: NonNegativeInt
    championName: str
    championTransform: NonNegativeInt
    commandPings: NonNegativeInt
    consumablesPurchased: NonNegativeInt
    damageDealtToBuildings: NonNegativeInt
    damageDealtToEpicMonsters: NonNegativeInt
    damageDealtToObjectives: NonNegativeInt
    damageDealtToTurrets: NonNegativeInt
    damageSelfMitigated: NonNegativeInt
    dangerPings: NonNegativeInt
    deaths: NonNegativeInt
    detectorWardsPlaced: NonNegativeInt
    doubleKills: NonNegativeInt
    dragonKills: NonNegativeInt
    eligibleForProgression: bool
    enemyMissingPings: NonNegativeInt
    enemyVisionPings: NonNegativeInt
    firstBloodAssist: bool
    firstBloodKill: bool
    firstTowerAssist: bool
    firstTowerKill: bool
    gameEndedInEarlySurrender: bool
    gameEndedInSurrender: bool
    getBackPings: NonNegativeInt
    goldEarned: NonNegativeInt
    goldSpent: NonNegativeInt
    holdPings: NonNegativeInt
    individualPosition: IndividualPosition
    inhibitorKills: NonNegativeInt
    inhibitorTakedowns: NonNegativeInt
    inhibitorsLost: NonNegativeInt
    item0: NonNegativeInt
    item1: NonNegativeInt
    item2: NonNegativeInt
    item3: NonNegativeInt
    item4: NonNegativeInt
    item5: NonNegativeInt
    item6: NonNegativeInt
    itemsPurchased: NonNegativeInt
    killingSprees: NonNegativeInt
    kills: NonNegativeInt
    lane: Lane
    largestCriticalStrike: NonNegativeInt
    largestKillingSpree: NonNegativeInt
    largestMultiKill: NonNegativeInt
    longestTimeSpentLiving: NonNegativeInt
    magicDamageDealt: NonNegativeInt
    magicDamageDealtToChampions: NonNegativeInt
    magicDamageTaken: NonNegativeInt
    missions: Missions
    needVisionPings: NonNegativeInt
    neutralMinionsKilled: NonNegativeInt
    nexusKills: NonNegativeInt
    nexusLost: NonNegativeInt
    nexusTakedowns: NonNegativeInt
    objectivesStolen: NonNegativeInt
    objectivesStolenAssists: NonNegativeInt
    onMyWayPings: NonNegativeInt
    participantId: NonNegativeInt
    pentaKills: NonNegativeInt
    perks: Perks
    physicalDamageDealt: NonNegativeInt
    physicalDamageDealtToChampions: NonNegativeInt
    physicalDamageTaken: NonNegativeInt
    placement: NonNegativeInt
    playerAugment1: NonNegativeInt
    playerAugment2: NonNegativeInt
    playerAugment3: NonNegativeInt
    playerAugment4: NonNegativeInt
    playerAugment5: NonNegativeInt
    playerAugment6: NonNegativeInt
    playerSubteamId: NonNegativeInt
    profileIcon: NonNegativeInt
    pushPings: NonNegativeInt
    puuid: str
    quadraKills: NonNegativeInt
    retreatPings: NonNegativeInt
    riotIdGameName: str
    riotIdTagline: str
    role: Role | None
    roleBoundItem: NonNegativeInt
    sightWardsBoughtInGame: NonNegativeInt
    spell1Casts: NonNegativeInt
    spell2Casts: NonNegativeInt
    spell3Casts: NonNegativeInt
    spell4Casts: NonNegativeInt
    subteamPlacement: NonNegativeInt
    summoner1Casts: NonNegativeInt
    summoner1Id: NonNegativeInt
    summoner2Casts: NonNegativeInt
    summoner2Id: NonNegativeInt
    summonerId: str
    summonerLevel: NonNegativeInt
    summonerName: str
    teamEarlySurrendered: bool
    teamId: TeamID
    teamPosition: TeamPosition
    timeCCingOthers: NonNegativeInt
    timePlayed: NonNegativeInt
    totalAllyJungleMinionsKilled: NonNegativeInt
    totalDamageDealt: NonNegativeInt
    totalDamageDealtToChampions: NonNegativeInt
    totalDamageShieldedOnTeammates: NonNegativeInt
    totalDamageTaken: NonNegativeInt
    totalEnemyJungleMinionsKilled: NonNegativeInt
    totalHeal: NonNegativeInt
    totalHealsOnTeammates: NonNegativeInt
    totalMinionsKilled: NonNegativeInt
    totalTimeCCDealt: NonNegativeInt
    totalTimeSpentDead: NonNegativeInt
    totalUnitsHealed: NonNegativeInt
    tripleKills: NonNegativeInt
    trueDamageDealt: NonNegativeInt
    trueDamageDealtToChampions: NonNegativeInt
    trueDamageTaken: NonNegativeInt
    turretKills: NonNegativeInt
    turretTakedowns: NonNegativeInt
    turretsLost: NonNegativeInt
    unrealKills: NonNegativeInt
    visionClearedPings: NonNegativeInt
    visionScore: NonNegativeInt
    visionWardsBoughtInGame: NonNegativeInt
    wardsKilled: NonNegativeInt
    wardsPlaced: NonNegativeInt
    win: bool


class FeatState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    featState: NonNegativeInt


class Feats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    EPIC_MONSTER_KILL: FeatState
    FIRST_BLOOD: FeatState
    FIRST_TURRET: FeatState


class Ban(BaseModel):
    model_config = ConfigDict(extra="forbid")
    championId: NonNegativeInt
    pickTurn: PickTurn


class ObjectiveStat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first: bool
    kills: NonNegativeInt


class Objectives(BaseModel):
    model_config = ConfigDict(extra="forbid")
    atakhan: ObjectiveStat
    baron: ObjectiveStat
    champion: ObjectiveStat
    dragon: ObjectiveStat
    horde: ObjectiveStat
    inhibitor: ObjectiveStat
    riftHerald: ObjectiveStat
    tower: ObjectiveStat


class Team(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bans: list[Ban]
    feats: Feats
    objectives: Objectives
    teamId: TeamID
    win: bool


class Info(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endOfGameResult: EndOfGameResult
    gameCreation: NonNegativeInt
    gameDuration: NonNegativeInt
    gameEndTimestamp: NonNegativeInt
    gameId: NonNegativeInt
    gameMode: GameMode
    gameName: str
    gameStartTimestamp: NonNegativeInt
    gameType: GameType
    gameVersion: str
    mapId: MapID
    participants: list[Participant]
    platformId: PlatformID
    queueId: QueueID
    teams: list[Team]
    tournamentCode: str


if __name__ == "__main__":
    path = Path("non-timeline.example.json")

    def load_dummy_non_timeline():
        with path.open("r") as f:
            data = json.load(f)

        return data

    data = load_dummy_non_timeline()
    validated_data = NonTimeline.model_validate(data)
