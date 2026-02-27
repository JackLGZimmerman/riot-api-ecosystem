from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

ChallengeValue = (
    NonNegativeInt | float | list[NonNegativeInt] | list[float] | str | bool | None
)


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataVersion: str
    matchId: str
    participants: list[str]


class Challenges(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    x12AssistStreakCount: ChallengeValue = Field(default=None, alias="12AssistStreakCount")
    HealFromMapSources: ChallengeValue = None
    InfernalScalePickup: ChallengeValue = None
    PlayerBehavior: ChallengeValue = None
    SWARM_DefeatAatrox: ChallengeValue = None
    SWARM_DefeatBriar: ChallengeValue = None
    SWARM_DefeatMiniBosses: ChallengeValue = None
    SWARM_EvolveWeapon: ChallengeValue = None
    SWARM_Have3Passives: ChallengeValue = None
    SWARM_KillEnemy: ChallengeValue = None
    SWARM_PickupGold: ChallengeValue = None
    SWARM_ReachLevel50: ChallengeValue = None
    SWARM_Survive15Min: ChallengeValue = None
    SWARM_WinWith5EvolvedWeapons: ChallengeValue = None
    abilityUses: ChallengeValue = None
    acesBefore15Minutes: ChallengeValue = None
    alliedJungleMonsterKills: ChallengeValue = None
    baronBuffGoldAdvantageOverThreshold: ChallengeValue = None
    baronTakedowns: ChallengeValue = None
    blastConeOppositeOpponentCount: ChallengeValue = None
    bountyGold: ChallengeValue = None
    buffsStolen: ChallengeValue = None
    completeSupportQuestInTime: ChallengeValue = None
    controlWardTimeCoverageInRiverOrEnemyHalf: ChallengeValue = None
    controlWardsPlaced: ChallengeValue = None
    damagePerMinute: ChallengeValue = None
    damageTakenOnTeamPercentage: ChallengeValue = None
    dancedWithRiftHerald: ChallengeValue = None
    deathsByEnemyChamps: ChallengeValue = None
    dodgeSkillShotsSmallWindow: ChallengeValue = None
    doubleAces: ChallengeValue = None
    dragonTakedowns: ChallengeValue = None
    earliestBaron: ChallengeValue = None
    earliestDragonTakedown: ChallengeValue = None
    earliestElderDragon: ChallengeValue = None
    earlyLaningPhaseGoldExpAdvantage: ChallengeValue = None
    effectiveHealAndShielding: ChallengeValue = None
    elderDragonKillsWithOpposingSoul: ChallengeValue = None
    elderDragonMultikills: ChallengeValue = None
    enemyChampionImmobilizations: ChallengeValue = None
    enemyJungleMonsterKills: ChallengeValue = None
    epicMonsterKillsNearEnemyJungler: ChallengeValue = None
    epicMonsterKillsWithin30SecondsOfSpawn: ChallengeValue = None
    epicMonsterSteals: ChallengeValue = None
    epicMonsterStolenWithoutSmite: ChallengeValue = None
    firstTurretKilled: ChallengeValue = None
    firstTurretKilledTime: ChallengeValue = None
    fasterSupportQuestCompletion: ChallengeValue = None
    fastestLegendary: ChallengeValue = None
    fistBumpParticipation: ChallengeValue = None
    flawlessAces: ChallengeValue = None
    fullTeamTakedown: ChallengeValue = None
    gameLength: ChallengeValue = None
    getTakedownsInAllLanesEarlyJungleAsLaner: ChallengeValue = None
    goldPerMinute: ChallengeValue = None
    hadOpenNexus: ChallengeValue = None
    hadAfkTeammate: ChallengeValue = None
    highestChampionDamage: ChallengeValue = None
    highestCrowdControlScore: ChallengeValue = None
    highestWardKills: ChallengeValue = None
    immobilizeAndKillWithAlly: ChallengeValue = None
    initialBuffCount: ChallengeValue = None
    initialCrabCount: ChallengeValue = None
    jungleCsBefore10Minutes: ChallengeValue = None
    junglerKillsEarlyJungle: ChallengeValue = None
    junglerTakedownsNearDamagedEpicMonster: ChallengeValue = None
    kTurretsDestroyedBeforePlatesFall: ChallengeValue = None
    kda: ChallengeValue = None
    killAfterHiddenWithAlly: ChallengeValue = None
    killParticipation: ChallengeValue = None
    killedChampTookFullTeamDamageSurvived: ChallengeValue = None
    killingSprees: ChallengeValue = None
    killsNearEnemyTurret: ChallengeValue = None
    killsOnLanersEarlyJungleAsJungler: ChallengeValue = None
    killsOnOtherLanesEarlyJungleAsLaner: ChallengeValue = None
    killsOnRecentlyHealedByAramPack: ChallengeValue = None
    killsUnderOwnTurret: ChallengeValue = None
    killsWithHelpFromEpicMonster: ChallengeValue = None
    knockEnemyIntoTeamAndKill: ChallengeValue = None
    landSkillShotsEarlyGame: ChallengeValue = None
    laneMinionsFirst10Minutes: ChallengeValue = None
    laningPhaseGoldExpAdvantage: ChallengeValue = None
    legendaryCount: ChallengeValue = None
    legendaryItemUsed: ChallengeValue = None
    lostAnInhibitor: ChallengeValue = None
    maxCsAdvantageOnLaneOpponent: ChallengeValue = None
    maxKillDeficit: ChallengeValue = None
    maxLevelLeadLaneOpponent: ChallengeValue = None
    mejaisFullStackInTime: ChallengeValue = None
    moreEnemyJungleThanOpponent: ChallengeValue = None
    multiKillOneSpell: ChallengeValue = None
    multiTurretRiftHeraldCount: ChallengeValue = None
    multikills: ChallengeValue = None
    multikillsAfterAggressiveFlash: ChallengeValue = None
    outerTurretExecutesBefore10Minutes: ChallengeValue = None
    outnumberedKills: ChallengeValue = None
    outnumberedNexusKill: ChallengeValue = None
    perfectDragonSoulsTaken: ChallengeValue = None
    perfectGame: ChallengeValue = None
    pickKillWithAlly: ChallengeValue = None
    playedChampSelectPosition: ChallengeValue = None
    poroExplosions: ChallengeValue = None
    quickCleanse: ChallengeValue = None
    quickFirstTurret: ChallengeValue = None
    quickSoloKills: ChallengeValue = None
    riftHeraldTakedowns: ChallengeValue = None
    saveAllyFromDeath: ChallengeValue = None
    scuttleCrabKills: ChallengeValue = None
    shortestTimeToAceFromFirstTakedown: ChallengeValue = None
    skillshotsDodged: ChallengeValue = None
    skillshotsHit: ChallengeValue = None
    snowballsHit: ChallengeValue = None
    soloBaronKills: ChallengeValue = None
    soloKills: ChallengeValue = None
    soloTurretsLategame: ChallengeValue = None
    stealthWardsPlaced: ChallengeValue = None
    survivedSingleDigitHpCount: ChallengeValue = None
    survivedThreeImmobilizesInFight: ChallengeValue = None
    takedownOnFirstTurret: ChallengeValue = None
    takedowns: ChallengeValue = None
    takedownsAfterGainingLevelAdvantage: ChallengeValue = None
    takedownsBeforeJungleMinionSpawn: ChallengeValue = None
    takedownsFirstXMinutes: ChallengeValue = None
    takedownsInAlcove: ChallengeValue = None
    takedownsInEnemyFountain: ChallengeValue = None
    teleportTakedowns: ChallengeValue = None
    teamBaronKills: ChallengeValue = None
    teamDamagePercentage: ChallengeValue = None
    teamElderDragonKills: ChallengeValue = None
    teamRiftHeraldKills: ChallengeValue = None
    thirdInhibitorDestroyedTime: ChallengeValue = None
    tookLargeDamageSurvived: ChallengeValue = None
    turretPlatesTaken: ChallengeValue = None
    turretTakedowns: ChallengeValue = None
    turretsTakenWithRiftHerald: ChallengeValue = None
    twentyMinionsIn3SecondsCount: ChallengeValue = None
    twoWardsOneSweeperCount: ChallengeValue = None
    unseenRecalls: ChallengeValue = None
    visionScoreAdvantageLaneOpponent: ChallengeValue = None
    visionScorePerMinute: ChallengeValue = None
    voidMonsterKill: ChallengeValue = None
    wardTakedowns: ChallengeValue = None
    wardTakedownsBefore20M: ChallengeValue = None
    wardsGuarded: ChallengeValue = None


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
    playerAugment1: NonNegativeInt | None = None
    playerAugment2: NonNegativeInt | None = None
    playerAugment3: NonNegativeInt | None = None
    playerAugment4: NonNegativeInt | None = None
    playerAugment5: NonNegativeInt | None = None
    playerAugment6: NonNegativeInt | None = None
    playerSubteamId: NonNegativeInt
    profileIcon: NonNegativeInt
    pushPings: NonNegativeInt
    puuid: str
    quadraKills: NonNegativeInt
    retreatPings: NonNegativeInt | None = None
    riotIdGameName: str
    riotIdTagline: str
    role: str | None
    roleBoundItem: int | None = None  # Schema drift, likely latest-season field behavior.
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
