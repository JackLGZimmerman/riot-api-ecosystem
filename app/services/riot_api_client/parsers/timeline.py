from __future__ import annotations

from dataclasses import dataclass, field
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

from app.services.riot_api_client.parsers.base_parsers import EventParser, InfoParser
from app.services.riot_api_client.parsers.models.timeline import (
    DamageInstance,
    EventChampionKill,
    Frame,
    Timeline,
)


def champion_kill_event_id(
    *,
    gameId: int,
    timestamp: int,
    killerId: int,
    victimId: int,
) -> str:
    return f"{gameId}:{timestamp}:{killerId}:{victimId}"


class ParticipantStatsRow(TypedDict):
    frame_timestamp: NonNegativeInt
    participantId: NonNegativeInt

    abilityHaste: NonNegativeInt
    abilityPower: NonNegativeInt
    armor: NonNegativeInt
    attackDamage: NonNegativeInt
    attackSpeed: NonNegativeInt
    ccReduction: NonNegativeInt
    cooldownReduction: NonNegativeInt
    health: NonNegativeInt
    healthMax: NonNegativeInt
    healthRegen: NonNegativeInt
    magicResist: NonNegativeInt
    movementSpeed: NonNegativeInt
    power: NonNegativeInt
    powerMax: NonNegativeInt
    powerRegen: NonNegativeInt
    payload: dict[str, NonNegativeInt]

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
    position_x: int
    position_y: int
    timeEnemySpentControlled: NonNegativeInt
    totalGold: NonNegativeInt
    xp: NonNegativeInt


class ParticipantStatsParser:
    def parse(self, frames: list[Frame]) -> list[ParticipantStatsRow]:
        rows: list[ParticipantStatsRow] = []

        for frame in frames:
            frame_timestamp: NonNegativeInt = (frame.timestamp // 10_000) * 10_000

            for pf in frame.participantFrames.root.values():
                champion_stats = pf.championStats.model_dump()
                payload = {
                    "armorPen": champion_stats.pop("armorPen"),
                    "armorPenPercent": champion_stats.pop("armorPenPercent"),
                    "bonusArmorPenPercent": champion_stats.pop("bonusArmorPenPercent"),
                    "bonusMagicPenPercent": champion_stats.pop("bonusMagicPenPercent"),
                    "magicPen": champion_stats.pop("magicPen"),
                    "magicPenPercent": champion_stats.pop("magicPenPercent"),
                    "lifesteal": champion_stats.pop("lifesteal"),
                    "omnivamp": champion_stats.pop("omnivamp"),
                    "physicalVamp": champion_stats.pop("physicalVamp"),
                    "spellVamp": champion_stats.pop("spellVamp"),
                }
                row_dict: dict[str, Any] = {
                    "frame_timestamp": frame_timestamp,
                    "participantId": pf.participantId,
                    **champion_stats,
                    "payload": payload,
                    "currentGold": pf.currentGold,
                    **pf.damageStats.model_dump(),
                    "goldPerSecond": pf.goldPerSecond,
                    "jungleMinionsKilled": pf.jungleMinionsKilled,
                    "level": pf.level,
                    "minionsKilled": pf.minionsKilled,
                    "position_x": int(pf.position.x) if pf.position else 0,
                    "position_y": int(pf.position.y) if pf.position else 0,
                    "timeEnemySpentControlled": pf.timeEnemySpentControlled,
                    "totalGold": pf.totalGold,
                    "xp": pf.xp,
                }

                rows.append(cast(ParticipantStatsRow, row_dict))

        return rows


class TimelineEventRowBase(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    timestamp: int


class BuildingKillRow(TimelineEventRowBase):
    type: Literal["BUILDING_KILL"]
    bounty: NonNegativeInt
    buildingType: str
    killerId: int
    laneType: str
    position_x: int
    position_y: int
    teamId: NonNegativeInt
    towerType: str


class ChampionKillRow(TimelineEventRowBase):
    type: Literal["CHAMPION_KILL"]
    champion_kill_event_id: str
    killerId: int
    victimId: int
    bounty: int
    killStreakLength: int
    shutdownBounty: int
    position_x: int
    position_y: int


class ChampionKillDamageInstanceRow(DamageInstance):
    gameId: NonNegativeInt
    frame_timestamp: int
    timestamp: int
    champion_kill_event_id: str
    direction: Literal["DEALT", "RECEIVED"]
    idx: NonNegativeInt


class ChampionSpecialKillRow(TimelineEventRowBase):
    type: Literal["CHAMPION_SPECIAL_KILL"]
    killType: str
    killerId: int
    position_x: int
    position_y: int
    multiKillLength: int


class DragonSoulGivenRow(TimelineEventRowBase):
    type: Literal["DRAGON_SOUL_GIVEN"]
    name: str
    teamId: int


class EliteMonsterKillRow(TimelineEventRowBase):
    type: Literal["ELITE_MONSTER_KILL"]
    assistingParticipantIds: list[int]
    bounty: int
    killerId: int
    killerTeamId: int
    monsterSubType: str
    monsterType: str
    position_x: int
    position_y: int


class RareEventRow(TimelineEventRowBase):
    type: Literal[
        "WARD_KILL",
        "WARD_PLACED",
        "GAME_END",
        "ITEM_DESTROYED",
        "ITEM_PURCHASED",
        "ITEM_SOLD",
        "ITEM_UNDO",
        "LEVEL_UP",
        "PAUSE_END",
        "SKILL_LEVEL_UP",
    ]
    payload: dict[str, Any]


class TurretPlateDestroyedRow(TimelineEventRowBase):
    type: Literal["TURRET_PLATE_DESTROYED"]
    killerId: int
    laneType: str
    position_x: int
    position_y: int
    teamId: int


RowT = TypeVar("RowT")


class EventTypeParser(Generic[RowT]):
    EVENT_TYPE: ClassVar[str]

    def parse(self, frames: list[Frame], gameId: int) -> list[RowT]:
        rows: list[RowT] = []

        for frame in frames:
            frame_timestamp = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                row: dict[str, Any] = {
                    **e,
                    "frame_timestamp": frame_timestamp,
                    "gameId": gameId,
                }

                pos = e.get("position")
                if pos is not None:
                    if isinstance(pos, dict):
                        row["position_x"] = pos["x"]
                        row["position_y"] = pos["y"]
                    else:
                        row["position_x"] = pos.x
                        row["position_y"] = pos.y

                    row.pop("position", None)

                rows.append(cast(RowT, row))

        return rows


class EventPayloadParser(Generic[RowT]):
    EVENT_TYPES: ClassVar[set[str]]

    def parse(self, frames: list[Frame], gameId: int) -> list[RowT]:
        rows: list[RowT] = []

        for frame in frames:
            frame_timestamp = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                event_type = e["type"]
                if event_type not in self.EVENT_TYPES:
                    continue

                payload = {
                    k: v
                    for k, v in e.items()
                    if k not in {"type", "timestamp", "gameId"}
                }
                rows.append(
                    cast(
                        RowT,
                        {
                            "gameId": gameId,
                            "frame_timestamp": frame_timestamp,
                            "type": event_type,
                            "timestamp": e["timestamp"],
                            "payload": payload,
                        },
                    )
                )

        return rows


class BuildingKillParser(EventTypeParser[BuildingKillRow]):
    EVENT_TYPE = "BUILDING_KILL"


class ChampionKillParser(EventTypeParser[ChampionKillRow]):
    EVENT_TYPE = "CHAMPION_KILL"

    def parse(self, frames: list[Frame], gameId: int) -> list[ChampionKillRow]:
        rows: list[ChampionKillRow] = []

        for frame in frames:
            frame_ts = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                e2: dict[str, Any] = dict(e)
                e2.pop("victimDamageDealt", None)
                e2.pop("victimDamageReceived", None)

                row: dict[str, Any] = {
                    **e2,
                    "frame_timestamp": frame_ts,
                    "gameId": gameId,
                }

                pos = e2.get("position")
                if pos is not None:
                    if isinstance(pos, dict):
                        row["position_x"] = pos["x"]
                        row["position_y"] = pos["y"]
                    else:
                        row["position_x"] = pos.x
                        row["position_y"] = pos.y

                    row.pop("position", None)

                rows.append(cast(ChampionKillRow, row))

        return rows


class ChampionKillDamageInstanceParser:
    KEY: ClassVar[Literal["victimDamageDealt", "victimDamageReceived"]]
    DIRECTION: ClassVar[Literal["DEALT", "RECEIVED"]]

    def parse(
        self, frames: list["Frame"], gameId: int
    ) -> list[ChampionKillDamageInstanceRow]:
        rows: list[ChampionKillDamageInstanceRow] = []

        for frame in frames:
            frame_ts = int((frame.timestamp // 10_000) * 10_000)

            for e in frame.events:
                if e["type"] != "CHAMPION_KILL":
                    continue

                ck = cast(EventChampionKill, e)

                cid = champion_kill_event_id(
                    gameId=gameId,
                    timestamp=int(ck["timestamp"]),
                    killerId=int(ck["killerId"]),
                    victimId=int(ck["victimId"]),
                )

                instances = cast(list[DamageInstance], ck.get(self.KEY, []))
                for idx, d in enumerate(instances):
                    rows.append(
                        {
                            **d,
                            "gameId": gameId,
                            "frame_timestamp": frame_ts,
                            "timestamp": e["timestamp"],
                            "direction": self.DIRECTION,
                            "champion_kill_event_id": cid,
                            "idx": idx,
                        }
                    )

        return rows


class VictimDamageDealtParser(ChampionKillDamageInstanceParser):
    KEY = "victimDamageDealt"
    DIRECTION = "DEALT"


class VictimDamageReceivedParser(ChampionKillDamageInstanceParser):
    KEY = "victimDamageReceived"
    DIRECTION = "RECEIVED"


class ChampionSpecialKillParser(EventTypeParser[ChampionSpecialKillRow]):
    EVENT_TYPE = "CHAMPION_SPECIAL_KILL"


class DragonSoulGivenParser(EventTypeParser[DragonSoulGivenRow]):
    EVENT_TYPE = "DRAGON_SOUL_GIVEN"


class EliteMonsterKillParser(EventTypeParser[EliteMonsterKillRow]):
    EVENT_TYPE = "ELITE_MONSTER_KILL"


class RareEventParser(EventPayloadParser[RareEventRow]):
    EVENT_TYPES = {
        "WARD_KILL",
        "WARD_PLACED",
        "GAME_END",
        "ITEM_DESTROYED",
        "ITEM_PURCHASED",
        "ITEM_SOLD",
        "ITEM_UNDO",
        "LEVEL_UP",
        "PAUSE_END",
        "SKILL_LEVEL_UP",
    }


class TurretPlateDestroyedParser(EventTypeParser[TurretPlateDestroyedRow]):
    EVENT_TYPE = "TURRET_PLATE_DESTROYED"


@dataclass
class TimelineTables:
    participantStats: list[ParticipantStatsRow]

    buildingKill: list[BuildingKillRow]
    championKill: list[ChampionKillRow]
    championSpecialKill: list[ChampionSpecialKillRow]
    dragonSoulGiven: list[DragonSoulGivenRow]
    eliteMonsterKill: list[EliteMonsterKillRow]
    payloadEvents: list[RareEventRow]

    turretPlateDestroyed: list[TurretPlateDestroyedRow]

    championKillVictimDamageDealt: list[ChampionKillDamageInstanceRow]
    championKillVictimDamageReceived: list[ChampionKillDamageInstanceRow]


@dataclass(frozen=True)
class MatchDataTimelineParsingOrchestrator:
    participantStats: InfoParser[list[Frame], list[ParticipantStatsRow]] = field(
        default_factory=ParticipantStatsParser
    )

    buildingKill: EventParser[list[Frame], list[BuildingKillRow]] = field(
        default_factory=BuildingKillParser
    )
    championKill: EventParser[list[Frame], list[ChampionKillRow]] = field(
        default_factory=ChampionKillParser
    )
    championSpecialKill: EventParser[list[Frame], list[ChampionSpecialKillRow]] = field(
        default_factory=ChampionSpecialKillParser
    )
    dragonSoulGiven: EventParser[list[Frame], list[DragonSoulGivenRow]] = field(
        default_factory=DragonSoulGivenParser
    )
    eliteMonsterKill: EventParser[list[Frame], list[EliteMonsterKillRow]] = field(
        default_factory=EliteMonsterKillParser
    )
    payloadEvents: EventParser[list[Frame], list[RareEventRow]] = field(
        default_factory=RareEventParser
    )

    turretPlateDestroyed: EventParser[list[Frame], list[TurretPlateDestroyedRow]] = (
        field(default_factory=TurretPlateDestroyedParser)
    )

    championKillVictimDamageDealt: EventParser[
        list[Frame], list[ChampionKillDamageInstanceRow]
    ] = field(default_factory=VictimDamageDealtParser)

    championKillVictimDamageReceived: EventParser[
        list[Frame], list[ChampionKillDamageInstanceRow]
    ] = field(default_factory=VictimDamageReceivedParser)

    def run(self, raw: dict[str, Any]) -> TimelineTables:
        try:
            tl = Timeline.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"raw did not match Timeline schema: {e}") from e

        info = tl.info
        frames = info.frames
        gameId = int(info.gameId)

        return TimelineTables(
            participantStats=self.participantStats.parse(frames),
            buildingKill=self.buildingKill.parse(frames, gameId),
            championKill=self.championKill.parse(frames, gameId),
            championSpecialKill=self.championSpecialKill.parse(frames, gameId),
            dragonSoulGiven=self.dragonSoulGiven.parse(frames, gameId),
            eliteMonsterKill=self.eliteMonsterKill.parse(frames, gameId),
            payloadEvents=self.payloadEvents.parse(frames, gameId),
            turretPlateDestroyed=self.turretPlateDestroyed.parse(frames, gameId),
            championKillVictimDamageDealt=self.championKillVictimDamageDealt.parse(
                frames, gameId
            ),
            championKillVictimDamageReceived=self.championKillVictimDamageReceived.parse(
                frames, gameId
            ),
        )
