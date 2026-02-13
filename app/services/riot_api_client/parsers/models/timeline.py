from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, RootModel


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: NonNegativeInt
    y: NonNegativeInt


class EventBase(TypedDict):
    frame_timestamp: NonNegativeInt
    timestamp: int


class UnknownEvent(EventBase):
    type: Literal["UNKNOWN"]


class EventItemPurchased(EventBase):
    type: Literal["ITEM_PURCHASED"]
    participantId: int
    itemId: int


class EventItemUndo(EventBase):
    type: Literal["ITEM_UNDO"]
    afterId: NonNegativeInt
    beforeId: NonNegativeInt
    goldGain: NonNegativeInt
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


class EventLevelUp(EventBase):
    type: Literal["LEVEL_UP"]
    level: NonNegativeInt
    participantId: NonNegativeInt


class EventItemDestroyed(EventBase):
    type: Literal["ITEM_DESTROYED"]
    itemId: NonNegativeInt
    participantId: int


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



class EventChampionKill(EventBase):
    type: Literal["CHAMPION_KILL"]
    bounty: NonNegativeInt
    killStreakLength: NonNegativeInt
    killerId: int
    position: Position
    shutdownBounty: NonNegativeInt
    victimDamageDealt: list[DamageInstance]
    victimDamageReceived: list[DamageInstance]
    victimId: int


class EventChampionSpecialKill(EventBase):
    type: Literal["CHAMPION_SPECIAL_KILL"]
    killType: str
    killerId: int
    position: Position


class EventTurretPlateDestroyed(EventBase):
    type: Literal["TURRET_PLATE_DESTROYED"]
    killerId: NonNegativeInt
    laneType: str
    position: Position
    teamId: NonNegativeInt


class EventBuildingKill(EventBase):
    type: Literal["BUILDING_KILL"]
    bounty: NonNegativeInt
    buildingType: str
    killerId: NonNegativeInt
    laneType: str
    position: Position
    teamId: NonNegativeInt


Event = Annotated[
    EventItemPurchased
    | EventItemUndo
    | EventSkillLevelUp
    | EventWardPlaced
    | EventLevelUp
    | EventItemDestroyed
    | EventChampionKill
    | EventChampionSpecialKill
    | EventTurretPlateDestroyed
    | EventBuildingKill
    | UnknownEvent,
    Field(discriminator="type"),
]


class ChampionStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    abilityHaste: NonNegativeInt
    abilityPower: NonNegativeInt
    armor: NonNegativeInt
    armorPen: NonNegativeInt
    armorPenPercent: NonNegativeInt
    attackDamage: NonNegativeInt
    attackSpeed: NonNegativeInt
    bonusArmorPenPercent: NonNegativeInt
    bonusMagicPenPercent: NonNegativeInt
    ccReduction: NonNegativeInt
    cooldownReduction: NonNegativeInt
    health: NonNegativeInt
    healthMax: NonNegativeInt
    healthRegen: NonNegativeInt
    lifesteal: NonNegativeInt
    magicPen: NonNegativeInt
    magicPenPercent: NonNegativeInt
    magicResist: NonNegativeInt
    movementSpeed: NonNegativeInt
    omnivamp: NonNegativeInt
    physicalVamp: NonNegativeInt
    power: NonNegativeInt
    powerMax: NonNegativeInt
    powerRegen: NonNegativeInt
    spellVamp: NonNegativeInt


class DamageStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
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
    model_config = ConfigDict(extra="forbid")
    championStats: ChampionStats
    currentGold: NonNegativeInt
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
    model_config = ConfigDict(extra="forbid")
    events: list[Event]
    participantFrames: ParticipantFrames
    timestamp: NonNegativeInt


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participantId: NonNegativeInt
    puuid: str


class Info(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endOfGameResult: str
    framePositiveInterval: NonNegativeInt
    frames: list[Frame]
    gameId: NonNegativeInt
    participants: list[Participant]


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataVersion: str
    matchId: str
    participants: list[str]


class Timeline(BaseModel):
    metadata: Metadata
    info: Info
