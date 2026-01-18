from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    TypedDict,
    TypeVar,
    cast,
)

from pydantic import (
    NonNegativeInt,
    ValidationError,
)

from app.services.riot_api_client.parsers.base_parsers import EventParser
from app.services.riot_api_client.parsers.models.timeline import (
    EventBase,
    Frame,
    Info,
    Metadata,
    Position,
    Timeline,
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


class TabulatedParticipantStats(TypedDict):
    frame_timestamp: NonNegativeInt
    participantId: NonNegativeInt

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

    currentGold: NonNegativeInt

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

    goldPerSecond: NonNegativeInt
    jungleMinionsKilled: NonNegativeInt
    level: NonNegativeInt
    minionsKilled: NonNegativeInt
    position: Position
    timeEnemySpentControlled: NonNegativeInt
    totalGold: NonNegativeInt
    xp: NonNegativeInt


class ParticipantStatsParser:
    def parse(self, frames: list[Frame]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for frame in frames:
            frame_timestamp: NonNegativeInt = frame.timestamp
            for pf in frame.participantFrames.root.values():
                rows.append(
                    {
                        "frame_timestamp": frame_timestamp,
                        **pf.championStats.model_dump(),
                        "currentGold": pf.currentGold,
                        **pf.damageStats.model_dump(),
                        "goldPerSecond": pf.goldPerSecond,
                        "jungleMinionsKilled": pf.jungleMinionsKilled,
                        "level": pf.level,
                        "minionsKilled": pf.minionsKilled,
                        "participantId": pf.participantId,
                        "position": pf.position,
                        "timeEnemySpentControlled": pf.timeEnemySpentControlled,
                        "totalGold": pf.totalGold,
                        "xp": pf.xp,
                    }
                )

        return rows


class BuildingKillPayload(TypedDict, total=False): ...


class BuildingKillRow(EventBase):
    type: Literal["BUILDING_KILL"]
    payload: BuildingKillPayload


class ChampionKillPayload(TypedDict, total=False): ...


class ChampionKillRow(EventBase):
    type: Literal["CHAMPION_KILL"]
    payload: ChampionKillPayload


class ChampionSpecialKillPayload(TypedDict, total=False): ...


class ChampionSpecialKillRow(EventBase):
    type: Literal["CHAMPION_SPECIAL_KILL"]
    payload: ChampionSpecialKillPayload


class DragonSoulGivenPayload(TypedDict, total=False): ...


class DragonSoulGivenRow(EventBase):
    type: Literal["DRAGON_SOUL_GIVEN"]
    payload: DragonSoulGivenPayload


class EliteMonsterKillPayload(TypedDict, total=False): ...


class EliteMonsterKillRow(EventBase):
    type: Literal["ELITE_MONSTER_KILL"]
    payload: EliteMonsterKillPayload


class GameEndPayload(TypedDict, total=False): ...


class GameEndRow(EventBase):
    type: Literal["GAME_END"]
    payload: GameEndPayload


class ItemDestroyedPayload(TypedDict, total=False): ...


class ItemDestroyedRow(EventBase):
    type: Literal["ITEM_DESTROYED"]
    payload: ItemDestroyedPayload


class ItemPurchasedPayload(TypedDict, total=False): ...


class ItemPurchasedRow(EventBase):
    type: Literal["ITEM_PURCHASED"]
    payload: ItemPurchasedPayload


class ItemSoldPayload(TypedDict, total=False): ...


class ItemSoldRow(EventBase):
    type: Literal["ITEM_SOLD"]
    payload: ItemSoldPayload


class ItemUndoPayload(TypedDict, total=False): ...


class ItemUndoRow(EventBase):
    type: Literal["ITEM_UNDO"]
    payload: ItemUndoPayload


class LevelUpPayload(TypedDict, total=False): ...


class LevelUpRow(EventBase):
    type: Literal["LEVEL_UP"]
    payload: LevelUpPayload


class PauseEndPayload(TypedDict, total=False): ...


class PauseEndRow(EventBase):
    type: Literal["PAUSE_END"]
    payload: PauseEndPayload


class SkillLevelUpPayload(TypedDict, total=False): ...


class SkillLevelUpRow(EventBase):
    type: Literal["SKILL_LEVEL_UP"]
    payload: SkillLevelUpPayload


class TurretPlateDestroyedPayload(TypedDict, total=False): ...


class TurretPlateDestroyedRow(EventBase):
    type: Literal["TURRET_PLATE_DESTROYED"]
    payload: TurretPlateDestroyedPayload


class WardKillPayload(TypedDict, total=False): ...


class WardKillRow(EventBase):
    type: Literal["WARD_KILL"]
    payload: WardKillPayload


class WardPlacedPayload(TypedDict, total=False): ...


class WardPlacedRow(EventBase):
    type: Literal["WARD_PLACED"]
    payload: WardPlacedPayload


RowT = TypeVar("RowT", bound=EventBase)


class EventTypeParser(Generic[RowT]):
    EVENT_TYPE: ClassVar[str]

    def parse(self, frames: list["Frame"]) -> list[RowT]:
        rows: list[RowT] = []

        for frame in frames:
            frame_timestamp: "NonNegativeInt" = frame.timestamp

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                type_: str = e["type"]
                rows.append(
                    cast(
                        RowT,
                        {
                            "frame_timestamp": frame_timestamp,
                            "type": type_,
                            "timestamp": e["timestamp"],
                            "payload": {
                                k: v
                                for k, v in e.items()
                                if k not in {"type", "timestamp"}
                            },
                        },
                    )
                )

        return rows


class BuildingKillParser(EventTypeParser[BuildingKillRow]):
    EVENT_TYPE = "BUILDING_KILL"


class ChampionKillParser(EventTypeParser[ChampionKillRow]):
    EVENT_TYPE = "CHAMPION_KILL"


class ChampionSpecialKillParser(EventTypeParser[ChampionSpecialKillRow]):
    EVENT_TYPE = "CHAMPION_SPECIAL_KILL"


class DragonSoulGivenParser(EventTypeParser[DragonSoulGivenRow]):
    EVENT_TYPE = "DRAGON_SOUL_GIVEN"


class EliteMonsterKillParser(EventTypeParser[EliteMonsterKillRow]):
    EVENT_TYPE = "ELITE_MONSTER_KILL"


class GameEndParser(EventTypeParser[GameEndRow]):
    EVENT_TYPE = "GAME_END"


class ItemDestroyedParser(EventTypeParser[ItemDestroyedRow]):
    EVENT_TYPE = "ITEM_DESTROYED"


class ItemPurchasedParser(EventTypeParser[ItemPurchasedRow]):
    EVENT_TYPE = "ITEM_PURCHASED"


class ItemSoldParser(EventTypeParser[ItemSoldRow]):
    EVENT_TYPE = "ITEM_SOLD"


class ItemUndoParser(EventTypeParser[ItemUndoRow]):
    EVENT_TYPE = "ITEM_UNDO"


class LevelUpParser(EventTypeParser[LevelUpRow]):
    EVENT_TYPE = "LEVEL_UP"


class PauseEndParser(EventTypeParser[PauseEndRow]):
    EVENT_TYPE = "PAUSE_END"


class SkillLevelUpParser(EventTypeParser[SkillLevelUpRow]):
    EVENT_TYPE = "SKILL_LEVEL_UP"


class TurretPlateDestroyedParser(EventTypeParser[TurretPlateDestroyedRow]):
    EVENT_TYPE = "TURRET_PLATE_DESTROYED"


class WardKillParser(EventTypeParser[WardKillRow]):
    EVENT_TYPE = "WARD_KILL"


class WardPlacedParser(EventTypeParser[WardPlacedRow]):
    EVENT_TYPE = "WARD_PLACED"


@dataclass
class TimelineTables:
    metadata: TabulatedMetadata
    participantStats: list[TabulatedParticipantStats]


@dataclass(frozen=True)
class MatchDataTimelineParsingOrchestrator:
    metadata: EventParser[Metadata, TabulatedMetadata]
    participantStats: EventParser[list[Frame], list[TabulatedParticipantStats]]

    def run(self, raw: dict[str, Any]) -> TimelineTables:
        try:
            nt = Timeline.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"raw did not match Timeline schema: {e}") from e

        metadata: Metadata = nt.metadata
        info: Info = nt.info
        frames: list[Frame] = info.frames

        gameId = info.gameId

        return TimelineTables(
            metadata=self.metadata.parse(metadata),
            participantStats=self.participantStats.parse(frames),
        )


if __name__ == "__main__":
    path = Path("non-timeline.example.json")

    def load_dummy_non_timeline():
        with path.open("r") as f:
            data = json.load(f)

        return data

    data = load_dummy_non_timeline()
    validated_data = Timeline.model_validate(**data)
