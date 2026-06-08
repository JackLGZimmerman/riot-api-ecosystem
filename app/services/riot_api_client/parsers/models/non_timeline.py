from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    create_model,
)

ChallengeValue = (
    NonNegativeInt | float | list[NonNegativeInt] | list[float] | str | bool | None
)


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataVersion: str
    matchId: str
    participants: list[str]


# Canonical ordered challenge field names — the single source of truth shared
# with the participant_challenges output TypedDict in parsers/non_timeline.py.
# Unmodelled keys must fail loudly (extra="forbid") so we can decide whether to
# add or ignore them.
CHALLENGE_FIELDS: tuple[str, ...] = (
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

# Field name -> wire alias for challenge keys that aren't valid identifiers.
CHALLENGE_ALIASES: dict[str, str] = {"x12AssistStreakCount": "12AssistStreakCount"}

# Challenge fields tabulated as list[int] rather than float | None.
CHALLENGE_LIST_FIELDS: frozenset[str] = frozenset({"legendaryItemUsed"})


def _challenge_field_definition(name: str) -> tuple[object, object]:
    alias = CHALLENGE_ALIASES.get(name)
    if alias is not None:
        return (ChallengeValue, Field(default=None, alias=alias))
    return (ChallengeValue, None)


Challenges = create_model(
    "Challenges",
    __config__=ConfigDict(extra="forbid", populate_by_name=True),
    **{name: _challenge_field_definition(name) for name in CHALLENGE_FIELDS},
)


class Missions(BaseModel):
    playerScore0: NonNegativeInt | None = None
    playerScore1: NonNegativeInt | None = None
    playerScore2: NonNegativeInt | None = None
    playerScore3: NonNegativeInt | None = None
    playerScore4: NonNegativeInt | None = None
    playerScore5: NonNegativeInt | None = None
    playerScore6: NonNegativeInt | None = None
    playerScore7: NonNegativeInt | None = None
    playerScore8: NonNegativeInt | None = None
    playerScore9: NonNegativeInt | None = None
    playerScore10: NonNegativeInt | None = None
    playerScore11: NonNegativeInt | None = None


class StatPerks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defense: NonNegativeInt
    flex: NonNegativeInt
    offense: NonNegativeInt


class PerkSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    perk: NonNegativeInt
    var1: int
    var2: int
    var3: int


class PerkStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    selections: list[PerkSelection]
    style: NonNegativeInt


class Perks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statPerks: StatPerks
    styles: list[PerkStyle]


class PlayerBehaviorData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    PlayerBehavior_IsHeroInCombat: NonNegativeInt | None = None


class Participant(BaseModel):
    model_config = ConfigDict(extra="ignore")
    PlayerScore0: NonNegativeInt | None = None
    PlayerScore1: NonNegativeInt | None = None
    PlayerScore2: NonNegativeInt | None = None
    PlayerScore3: NonNegativeInt | None = None
    PlayerScore4: NonNegativeInt | None = None
    PlayerScore5: NonNegativeInt | None = None
    PlayerScore6: NonNegativeInt | None = None
    PlayerScore7: NonNegativeInt | None = None
    PlayerScore8: NonNegativeInt | None = None
    PlayerScore9: NonNegativeInt | None = None
    PlayerScore10: NonNegativeInt | None = None
    PlayerScore11: NonNegativeInt | None = None
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
    damageDealtToEpicMonsters: int | None = None  # Schema drift, likely latest-season field behavior.
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
    gameEndedInIGNBSurrender: bool | None = None  # Schema drift, new IGNB surrender variant.
    gameEndedInSurrender: bool
    causedGameEndFromIGNBSurrender: bool | None = None
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
    playerAugment1: NonNegativeInt | None = None
    playerAugment2: NonNegativeInt | None = None
    playerAugment3: NonNegativeInt | None = None
    playerAugment4: NonNegativeInt | None = None
    playerAugment5: NonNegativeInt | None = None
    playerAugment6: NonNegativeInt | None = None
    PlayerBehavior: PlayerBehaviorData | None = None  # Schema drift, nullable nested participant behavior payload.
    playerSubteamId: NonNegativeInt
    positionAssignedByMatchmaking: str | None = None
    profileIcon: NonNegativeInt
    pushPings: NonNegativeInt
    puuid: str
    quadraKills: NonNegativeInt
    retreatPings: NonNegativeInt | None = None
    riotIdGameName: str
    riotIdTagline: str
    role: str | None
    roleBoundItem: int | None = None  # Schema drift, likely latest-season field behavior.
    selectedRolePreferences: str | None = None
    bountyLevel: int | None = None  # Schema drift, likely latest-season addition.
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
    teamIGNBSurrendered: bool | None = None  # Schema drift, new IGNB surrender variant.
    wasPremadeWithIGNBGameEndCauser: bool | None = None
    wasPremadeWithSevereTransgressor: bool | None = None
    wasSevereTransgressor: bool | None = None
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
    model_config = ConfigDict(extra="ignore")
    EPIC_MONSTER_KILL: FeatState
    FIRST_BLOOD: FeatState
    FIRST_TURRET: FeatState


class Ban(BaseModel):
    model_config = ConfigDict(extra="forbid")
    championId: int
    pickTurn: PositiveInt


class ObjectiveStat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first: bool
    kills: NonNegativeInt


class Objectives(BaseModel):
    model_config = ConfigDict(extra="ignore")
    atakhan: ObjectiveStat | None = None  # Schema drift, likely latest-season objective.
    baron: ObjectiveStat
    champion: ObjectiveStat
    dragon: ObjectiveStat
    horde: ObjectiveStat
    inhibitor: ObjectiveStat
    riftHerald: ObjectiveStat
    tower: ObjectiveStat


class Team(BaseModel):
    model_config = ConfigDict(extra="ignore")
    bans: list[Ban]
    feats: Feats | None = None  # Schema drift, can be absent in some latest-season payloads.
    objectives: Objectives
    teamId: PositiveInt
    win: bool


class Info(BaseModel):
    model_config = ConfigDict(extra="ignore")
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
    mapId: NonNegativeInt
    participants: list[Participant]
    platformId: str
    queueId: NonNegativeInt
    teams: list[Team]
    tournamentCode: str


class NonTimeline(BaseModel):
    metadata: Metadata
    info: Info
