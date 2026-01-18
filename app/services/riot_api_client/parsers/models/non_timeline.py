from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, RootModel

ChallengeValue = (
    NonNegativeInt | float | list[NonNegativeInt] | list[float] | str | bool | None
)


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataVersion: str
    matchId: str
    participants: list[str]


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
    PlayerScore2: NonNegativeInt
    PlayerScore3: NonNegativeInt
    PlayerScore4: NonNegativeInt
    PlayerScore5: NonNegativeInt
    PlayerScore6: NonNegativeInt
    PlayerScore7: NonNegativeInt
    PlayerScore8: NonNegativeInt
    PlayerScore9: NonNegativeInt
    PlayerScore10: NonNegativeInt
    PlayerScore11: NonNegativeInt
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
    individualPosition: str
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
    lane: str
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
    role: str | None
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
    teamId: PositiveInt
    teamPosition: str
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
    pickTurn: PositiveInt


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
    teamId: PositiveInt
    win: bool


class Info(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endOfGameResult: str
    gameCreation: NonNegativeInt
    gameDuration: NonNegativeInt
    gameEndTimestamp: NonNegativeInt
    gameId: NonNegativeInt
    gameMode: str
    gameName: str
    gameStartTimestamp: NonNegativeInt
    gameType: str
    gameVersion: str
    mapId: PositiveInt
    participants: list[Participant]
    platformId: str
    queueId: PositiveInt
    teams: list[Team]
    tournamentCode: str


class NonTimeline(BaseModel):
    metadata: Metadata
    info: Info
