from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    DamageInstance,
    EventChampionKill,
    Frame,
    Timeline,
)
from app.services.riot_api_client.parsers.schema_drift import (
    timeline as timeline_drift,
)

logger = logging.getLogger(__name__)


def champion_kill_event_id(
    *,
    matchId: int,
    timestamp: int,
    killerId: int,
    victimId: int,
) -> str:
    return f"{matchId}:{timestamp}:{killerId}:{victimId}"


class ParticipantStatsRow(TypedDict):
    matchId: NonNegativeInt
    frame_timestamp: NonNegativeInt
    participantId: NonNegativeInt

    abilityHaste: NonNegativeInt
    abilityPower: NonNegativeInt
    armor: int
    attackDamage: int
    attackSpeed: NonNegativeInt
    ccReduction: int
    cooldownReduction: NonNegativeInt
    health: NonNegativeInt
    healthMax: NonNegativeInt
    healthRegen: NonNegativeInt
    magicResist: int
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
    def parse(self, frames: list[Frame], matchId: int) -> list[ParticipantStatsRow]:
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
                    "matchId": matchId,
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
    matchId: int
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
    towerType: str | None


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
    matchId: NonNegativeInt
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
    multiKillLength: int | None


class DragonSoulGivenRow(TimelineEventRowBase):
    type: Literal["DRAGON_SOUL_GIVEN"]
    name: str
    teamId: int


class EliteMonsterKillRow(TimelineEventRowBase):
    type: Literal["ELITE_MONSTER_KILL"]
    assistingParticipantIds: list[int] | None
    bounty: int
    killerId: int
    killerTeamId: int
    monsterSubType: str | None
    monsterType: str
    position_x: int
    position_y: int


class RareEventRow(TimelineEventRowBase):
    type: Literal[
        "WARD_KILL",
        "WARD_PLACED",
        "GAME_END",
        "OBJECTIVE_BOUNTY_PRESTART",
        "OBJECTIVE_BOUNTY_FINISH",
        "FEAT_UPDATE",
        "CHAMPION_TRANSFORM",
        "ITEM_DESTROYED",
        "ITEM_PURCHASED",
        "ITEM_SOLD",
        "ITEM_UNDO",
        "LEVEL_UP",
        "PAUSE_END",
        "SKILL_LEVEL_UP",
        "UNKNOWN",
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

    def parse(self, frames: list[Frame], matchId: int) -> list[RowT]:
        rows: list[RowT] = []

        for frame in frames:
            frame_timestamp = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                row: dict[str, Any] = {
                    **e,
                    "frame_timestamp": frame_timestamp,
                    "matchId": matchId,
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

    def parse(self, frames: list[Frame], matchId: int) -> list[RowT]:
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
                    if k not in {"type", "timestamp", "matchId", "gameId"}
                }
                rows.append(
                    cast(
                        RowT,
                        {
                            "matchId": matchId,
                            "frame_timestamp": frame_timestamp,
                            "type": event_type,
                            "timestamp": e["timestamp"],
                            "payload": payload,
                        },
                    )
                )

        return rows


class ChampionKillParser(EventTypeParser[ChampionKillRow]):
    EVENT_TYPE = "CHAMPION_KILL"

    def parse(self, frames: list[Frame], matchId: int) -> list[ChampionKillRow]:
        rows: list[ChampionKillRow] = []

        for frame in frames:
            frame_ts = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                e2: dict[str, Any] = dict(e)
                e2.pop("victimDamageDealt", None)
                e2.pop("victimDamageReceived", None)
                e2.pop("victimTeamfightDamageDealt", None)
                e2.pop("victimTeamfightDamageReceived", None)

                row: dict[str, Any] = {
                    **e2,
                    "frame_timestamp": frame_ts,
                    "matchId": matchId,
                    "champion_kill_event_id": champion_kill_event_id(
                        matchId=matchId,
                        timestamp=int(e2["timestamp"]),
                        killerId=int(e2["killerId"]),
                        victimId=int(e2["victimId"]),
                    ),
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
    ALIAS_KEY: ClassVar[
        Literal["victimTeamfightDamageDealt", "victimTeamfightDamageReceived"]
    ]
    DIRECTION: ClassVar[Literal["DEALT", "RECEIVED"]]

    def parse(
        self, frames: list["Frame"], matchId: int
    ) -> list[ChampionKillDamageInstanceRow]:
        rows: list[ChampionKillDamageInstanceRow] = []

        for frame in frames:
            frame_ts = int((frame.timestamp // 10_000) * 10_000)

            for e in frame.events:
                if e["type"] != "CHAMPION_KILL":
                    continue

                ck = cast(EventChampionKill, e)

                cid = champion_kill_event_id(
                    matchId=matchId,
                    timestamp=int(ck["timestamp"]),
                    killerId=int(ck["killerId"]),
                    victimId=int(ck["victimId"]),
                )

                if self.KEY in ck:
                    instances = cast(list[DamageInstance], ck.get(self.KEY, []))
                else:
                    instances = cast(list[DamageInstance], ck.get(self.ALIAS_KEY, []))
                for idx, d in enumerate(instances):
                    rows.append(
                        {
                            **d,
                            "matchId": matchId,
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
    ALIAS_KEY = "victimTeamfightDamageDealt"
    DIRECTION = "DEALT"


class VictimDamageReceivedParser(ChampionKillDamageInstanceParser):
    KEY = "victimDamageReceived"
    ALIAS_KEY = "victimTeamfightDamageReceived"
    DIRECTION = "RECEIVED"


class ChampionSpecialKillParser(EventTypeParser[ChampionSpecialKillRow]):
    EVENT_TYPE = "CHAMPION_SPECIAL_KILL"

    def parse(self, frames: list[Frame], matchId: int) -> list[ChampionSpecialKillRow]:
        rows: list[ChampionSpecialKillRow] = []

        for frame in frames:
            frame_timestamp = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                row: dict[str, Any] = {
                    **e,
                    "multiKillLength": e.get("multiKillLength"),
                    "frame_timestamp": frame_timestamp,
                    "matchId": matchId,
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

                rows.append(cast(ChampionSpecialKillRow, row))

        return rows


class DragonSoulGivenParser(EventTypeParser[DragonSoulGivenRow]):
    EVENT_TYPE = "DRAGON_SOUL_GIVEN"


class EliteMonsterKillParser(EventTypeParser[EliteMonsterKillRow]):
    EVENT_TYPE = "ELITE_MONSTER_KILL"

    def parse(self, frames: list[Frame], matchId: int) -> list[EliteMonsterKillRow]:
        rows: list[EliteMonsterKillRow] = []

        for frame in frames:
            frame_timestamp = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                row: dict[str, Any] = {
                    **e,
                    "assistingParticipantIds": e.get("assistingParticipantIds", []),
                    "monsterSubType": e.get("monsterSubType"),
                    "frame_timestamp": frame_timestamp,
                    "matchId": matchId,
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

                rows.append(cast(EliteMonsterKillRow, row))

        return rows


class RareEventParser(EventPayloadParser[RareEventRow]):
    EVENT_TYPES = {
        "WARD_KILL",
        "WARD_PLACED",
        "GAME_END",
        "OBJECTIVE_BOUNTY_PRESTART",
        "OBJECTIVE_BOUNTY_FINISH",
        "FEAT_UPDATE",
        "CHAMPION_TRANSFORM",
        "ITEM_DESTROYED",
        "ITEM_PURCHASED",
        "ITEM_SOLD",
        "ITEM_UNDO",
        "LEVEL_UP",
        "PAUSE_END",
        "SKILL_LEVEL_UP",
        "UNKNOWN",
    }


class TurretPlateDestroyedParser(EventTypeParser[TurretPlateDestroyedRow]):
    EVENT_TYPE = "TURRET_PLATE_DESTROYED"


class BuildingKillParser(EventTypeParser[BuildingKillRow]):
    EVENT_TYPE = "BUILDING_KILL"

    def parse(self, frames: list[Frame], matchId: int) -> list[BuildingKillRow]:
        rows: list[BuildingKillRow] = []

        for frame in frames:
            frame_timestamp = (frame.timestamp // 10_000) * 10_000

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                row: dict[str, Any] = {
                    **e,
                    "towerType": e.get("towerType"),
                    "frame_timestamp": frame_timestamp,
                    "matchId": matchId,
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

                rows.append(cast(BuildingKillRow, row))

        return rows


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
    participantStats: EventParser[list[Frame], list[ParticipantStatsRow]] = field(
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

    @staticmethod
    def _drift_date() -> str:
        return datetime.now(tz=UTC).date().isoformat()

    def run(self, raw: dict[str, Any]) -> TimelineTables:
        metadata_raw = raw.get("metadata", {})
        match_id = (
            metadata_raw.get("matchId", "unknown")
            if isinstance(metadata_raw, dict)
            else "unknown"
        )
        drift_date = self._drift_date()
        timeline_drift(raw, match_id=match_id, drift_date=drift_date)

        try:
            tl = Timeline.model_validate(raw)
            info = tl.info
            frames = info.frames
            matchId = int(info.gameId)

            tables = TimelineTables(
                participantStats=self.participantStats.parse(frames, matchId),
                buildingKill=self.buildingKill.parse(frames, matchId),
                championKill=self.championKill.parse(frames, matchId),
                championSpecialKill=self.championSpecialKill.parse(frames, matchId),
                dragonSoulGiven=self.dragonSoulGiven.parse(frames, matchId),
                eliteMonsterKill=self.eliteMonsterKill.parse(frames, matchId),
                payloadEvents=self.payloadEvents.parse(frames, matchId),
                turretPlateDestroyed=self.turretPlateDestroyed.parse(frames, matchId),
                championKillVictimDamageDealt=self.championKillVictimDamageDealt.parse(
                    frames, matchId
                ),
                championKillVictimDamageReceived=self.championKillVictimDamageReceived.parse(
                    frames, matchId
                ),
            )
        except ValidationError as e:
            logger.warning(
                "SchemaValidation timeline match_id=%s date=%s errors=%s",
                match_id,
                drift_date,
                e.errors(include_input=False),
            )
            logger.warning(
                "Skipping timeline payload for match_id=%s due to validation errors "
                "during initial schema tuning.",
                match_id,
            )
            # TODO: Re-enable hard failure (raise ValueError) after initial schema
            # tuning is complete and drift has stabilized.
            return TimelineTables(
                participantStats=[],
                buildingKill=[],
                championKill=[],
                championSpecialKill=[],
                dragonSoulGiven=[],
                eliteMonsterKill=[],
                payloadEvents=[],
                turretPlateDestroyed=[],
                championKillVictimDamageDealt=[],
                championKillVictimDamageReceived=[],
            )
        return tables
