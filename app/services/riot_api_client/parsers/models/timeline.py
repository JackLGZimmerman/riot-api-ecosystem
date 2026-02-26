from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    RootModel,
)


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: NonNegativeInt
    y: NonNegativeInt


class EventBase(TypedDict):
    timestamp: int


class UnknownEventBase(EventBase):
    type: Literal["UNKNOWN"]


class UnknownEventOptional(TypedDict, total=False):
    originalType: str


class UnknownEvent(UnknownEventBase, UnknownEventOptional):
    pass


class EventItemPurchased(EventBase):
    type: Literal["ITEM_PURCHASED"]
    participantId: int
    itemId: int


class EventItemUndo(EventBase):
    type: Literal["ITEM_UNDO"]
    afterId: NonNegativeInt
    beforeId: NonNegativeInt
    goldGain: int
    participantId: int


class EventSkillLevelUp(EventBase):
    type: Literal["SKILL_LEVEL_UP"]
    levelUpType: str
    participantId: int
    skillSlot: int


class EventWardPlaced(EventBase):
    type: Literal["WARD_PLACED"]
    creatorId: NonNegativeInt
    wardType: str


class EventWardKillBase(EventBase):
    type: Literal["WARD_KILL"]
    killerId: int
    wardType: str


class EventWardKillOptional(TypedDict, total=False):
    creatorId: int


class EventWardKill(EventWardKillBase, EventWardKillOptional):
    pass


class EventLevelUp(EventBase):
    type: Literal["LEVEL_UP"]
    level: NonNegativeInt
    participantId: NonNegativeInt


class EventItemDestroyed(EventBase):
    type: Literal["ITEM_DESTROYED"]
    itemId: NonNegativeInt
    participantId: int


class EventItemSold(EventBase):
    type: Literal["ITEM_SOLD"]
    itemId: NonNegativeInt
    participantId: int


class EventGameEndBase(EventBase):
    type: Literal["GAME_END"]
    winningTeam: int
    realTimestamp: NonNegativeInt


class EventGameEndOptional(TypedDict, total=False):
    gameId: int


class EventGameEnd(EventGameEndBase, EventGameEndOptional):
    pass


class EventPauseEnd(EventBase):
    type: Literal["PAUSE_END"]
    realTimestamp: NonNegativeInt


class DamageInstance(TypedDict):
    basic: bool
    magicDamage: NonNegativeInt
    name: str
    participantId: int
    physicalDamage: NonNegativeInt
    spellName: str
    spellSlot: int
    trueDamage: NonNegativeInt
    type: str


class EventChampionKillBase(EventBase):
    type: Literal["CHAMPION_KILL"]
    bounty: NonNegativeInt
    killStreakLength: NonNegativeInt
    killerId: int
    position: Position
    shutdownBounty: NonNegativeInt
    victimId: int


class EventChampionKillOptional(TypedDict, total=False):
    victimDamageDealt: list[DamageInstance]
    victimDamageReceived: list[DamageInstance]
    victimTeamfightDamageDealt: list[DamageInstance]
    victimTeamfightDamageReceived: list[DamageInstance]
    assistingParticipantIds: list[int]


class EventChampionKill(EventChampionKillBase, EventChampionKillOptional):
    pass


class EventChampionSpecialKillBase(EventBase):
    type: Literal["CHAMPION_SPECIAL_KILL"]
    killType: str
    killerId: int
    position: Position


class EventChampionSpecialKillOptional(TypedDict, total=False):
    multiKillLength: int


class EventChampionSpecialKill(
    EventChampionSpecialKillBase, EventChampionSpecialKillOptional
):
    pass


class EventDragonSoulGiven(EventBase):
    type: Literal["DRAGON_SOUL_GIVEN"]
    name: str
    teamId: int


class EventEliteMonsterKillBase(EventBase):
    type: Literal["ELITE_MONSTER_KILL"]
    bounty: int
    killerId: int
    killerTeamId: int
    monsterType: str
    position: Position


class EventEliteMonsterKillOptional(TypedDict, total=False):
    assistingParticipantIds: list[int]
    monsterSubType: str


class EventEliteMonsterKill(EventEliteMonsterKillBase, EventEliteMonsterKillOptional):
    pass


class EventTurretPlateDestroyed(EventBase):
    type: Literal["TURRET_PLATE_DESTROYED"]
    killerId: NonNegativeInt
    laneType: str
    position: Position
    teamId: NonNegativeInt


class EventBuildingKillBase(EventBase):
    type: Literal["BUILDING_KILL"]
    bounty: NonNegativeInt
    buildingType: str
    killerId: NonNegativeInt
    laneType: str
    position: Position
    teamId: NonNegativeInt


class EventBuildingKillOptional(TypedDict, total=False):
    towerType: str
    assistingParticipantIds: list[int]


class EventBuildingKill(EventBuildingKillBase, EventBuildingKillOptional):
    pass


class EventObjectiveBountyPrestart(EventBase):
    type: Literal["OBJECTIVE_BOUNTY_PRESTART"]
    actualStartTime: NonNegativeInt
    teamId: int


class EventObjectiveBountyFinish(EventBase):
    type: Literal["OBJECTIVE_BOUNTY_FINISH"]
    teamId: int


class EventFeatUpdate(EventBase):
    type: Literal["FEAT_UPDATE"]
    featType: int
    featValue: int
    teamId: int


class EventChampionTransform(EventBase):
    type: Literal["CHAMPION_TRANSFORM"]
    participantId: int
    transformType: str


Event = Annotated[
    EventItemPurchased
    | EventItemUndo
    | EventSkillLevelUp
    | EventWardKill
    | EventWardPlaced
    | EventLevelUp
    | EventGameEnd
    | EventItemDestroyed
    | EventItemSold
    | EventPauseEnd
    | EventChampionKill
    | EventChampionSpecialKill
    | EventDragonSoulGiven
    | EventEliteMonsterKill
    | EventTurretPlateDestroyed
    | EventBuildingKill
    | EventObjectiveBountyPrestart
    | EventObjectiveBountyFinish
    | EventFeatUpdate
    | EventChampionTransform
    | UnknownEvent,
    Field(discriminator="type"),
]


class ChampionStats(BaseModel):
    model_config = ConfigDict(extra="ignore")
    abilityHaste: NonNegativeInt
    abilityPower: NonNegativeInt
    armor: int
    armorPen: NonNegativeInt
    armorPenPercent: NonNegativeInt
    attackDamage: NonNegativeInt
    attackSpeed: NonNegativeInt
    bonusArmorPenPercent: NonNegativeInt
    bonusMagicPenPercent: NonNegativeInt
    ccReduction: int
    cooldownReduction: NonNegativeInt
    health: NonNegativeInt
    healthMax: NonNegativeInt
    healthRegen: NonNegativeInt
    lifesteal: NonNegativeInt
    magicPen: NonNegativeInt
    magicPenPercent: NonNegativeInt
    magicResist: int
    movementSpeed: NonNegativeInt
    omnivamp: NonNegativeInt
    physicalVamp: NonNegativeInt
    power: NonNegativeInt
    powerMax: NonNegativeInt
    powerRegen: NonNegativeInt
    spellVamp: NonNegativeInt


class DamageStats(BaseModel):
    model_config = ConfigDict(extra="ignore")
    magicDamageDone: NonNegativeInt
    magicDamageDoneToChampions: NonNegativeInt
    magicDamageTaken: NonNegativeInt
    physicalDamageDone: NonNegativeInt
    physicalDamageDoneToChampions: NonNegativeInt
    physicalDamageTaken: NonNegativeInt
    totalDamageDone: NonNegativeInt
    totalDamageDoneToChampions: NonNegativeInt
    totalDamageTaken: NonNegativeInt
    trueDamageDone: NonNegativeInt
    trueDamageDoneToChampions: NonNegativeInt
    trueDamageTaken: NonNegativeInt


class ParticipantStats(BaseModel):
    model_config = ConfigDict(extra="ignore")
    championStats: ChampionStats
    currentGold: int
    damageStats: DamageStats
    goldPerSecond: NonNegativeInt
    jungleMinionsKilled: NonNegativeInt
    level: NonNegativeInt
    minionsKilled: NonNegativeInt
    participantId: int
    position: Position
    timeEnemySpentControlled: NonNegativeInt
    totalGold: NonNegativeInt
    xp: NonNegativeInt


class ParticipantFrames(RootModel[dict[NonNegativeInt, ParticipantStats]]):
    pass


class Frame(BaseModel):
    model_config = ConfigDict(extra="ignore")
    events: list[Event]
    participantFrames: ParticipantFrames
    timestamp: NonNegativeInt


class Participant(BaseModel):
    model_config = ConfigDict(extra="ignore")
    participantId: NonNegativeInt
    puuid: str


class Info(BaseModel):
    model_config = ConfigDict(extra="ignore")
    endOfGameResult: str
    framePositiveInterval: NonNegativeInt = Field(
        default=0,
        validation_alias=AliasChoices("framePositiveInterval", "frameInterval"),
    )
    frames: list[Frame]
    gameId: NonNegativeInt = Field(validation_alias=AliasChoices("matchId", "gameId"))
    participants: list[Participant]


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataVersion: str
    matchId: str
    participants: list[str]


class Timeline(BaseModel):
    metadata: Metadata
    info: Info
