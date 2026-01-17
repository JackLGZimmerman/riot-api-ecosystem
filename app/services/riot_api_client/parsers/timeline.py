from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, RootModel, NonNegativeInt

Raw_contra = TypeVar("Raw_contra", bound=BaseModel, contravariant=True)
Out_co = TypeVar("Out_co", bound=BaseModel, covariant=True)
Out = TypeVar("Out", bound=BaseModel)


class EventParser(Protocol[Raw_contra, Out_co]):
    def parse(self, raw: Raw_contra) -> Out_co: ...


class Orchestrator(Protocol[Raw_contra, Out_co]):
    def run(self, raw: Raw_contra) -> Out_co: ...


EndOfGameResult = Literal["GameComplete"]
EventValues = NonNegativeInt | str | dict[str, Any]
ParticipantID = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class MatchMetadata(BaseModel):
    matchId: str


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participantId: NonNegativeInt
    puuid: str


class Event(RootModel[EventValues]):
    pass


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


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: NonNegativeInt
    y: NonNegativeInt


class ParticipantStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    championStats: ChampionStats
    currentGold: str
    damageStats: DamageStats
    goldPerSecond: NonNegativeInt
    jungleMinionsKilled: NonNegativeInt
    level: NonNegativeInt
    minionsKilled: NonNegativeInt
    participantId: ParticipantID
    position: Position
    timeEnemySpentControlled: NonNegativeInt
    totalGold: NonNegativeInt
    xp: NonNegativeInt


class ParticipantFrames(RootModel[dict[NonNegativeInt, ParticipantStats]]):
    pass


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: str
    participantFrames: str
    timestamp: NonNegativeInt


class Info(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endOfGameResult: EndOfGameResult
    framePositiveInterval: NonNegativeInt
    Frames: list[Frame]
    gameId: NonNegativeInt
    participants: list[Participant]


class Timeline(BaseModel):
    metadata: Metadata
    info: Info


class MetadataParser:
    def parse(self, raw: Timeline) -> MatchMetadata:
        return MatchMetadata(matchId=str(raw.metadata.matchId))


@dataclass(frozen=True)
class MatchDataTimelineParsingOrchestrator(Generic[Out]):
    metadata: EventParser[Timeline, Out]

    def run(self, raw: Timeline) -> Out:
        return self.metadata.parse(raw)


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataVersion: str
    matchId: str
    participants: list[str]


if __name__ == "__main__":
    path = Path("non-timeline.example.json")

    def load_dummy_non_timeline():
        with path.open("r") as f:
            data = json.load(f)

        return data

    data = load_dummy_non_timeline()
    validated_data = Timeline.model_validate(**data)

    # _orch = MatchDataNonTimelineParsingOrchestrator(metadata=MetadataParser())
    # orch_as_base: Orchestrator[NonTimeline, BaseModel] = _orch
