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
    EventBase,
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
    position_x: int
    position_y: int
    timeEnemySpentControlled: NonNegativeInt
    totalGold: NonNegativeInt
    xp: NonNegativeInt


class ParticipantStatsParser:
    def parse(self, frames: list[Frame]) -> list[ParticipantStatsRow]:
        rows: list[ParticipantStatsRow] = []

        for frame in frames:
            frame_timestamp: NonNegativeInt = frame.timestamp

            for pf in frame.participantFrames.root.values():
                row_dict: dict[str, Any] = {
                    "frame_timestamp": frame_timestamp,
                    "participantId": pf.participantId,
                    **pf.championStats.model_dump(),
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


class BuildingKillRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["BUILDING_KILL"]
    timestamp: int
    bounty: NonNegativeInt
    buildingType: str
    killerId: int
    laneType: str
    position_x: int
    position_y: int
    teamId: NonNegativeInt
    towerType: str


class ChampionKillRow(TypedDict):
    gameId: int
    frame_timestamp: int
    type: Literal["CHAMPION_SPECIAL_KILL"]
    timestamp: int
    champion_kill_event_id: str
    killerId: int
    victimId: int
    bounty: int
    killStreakLength: int
    shutdownBounty: int
    position: dict[str, int]


class ChampionKillDamageInstanceRow(DamageInstance):
    gameId: NonNegativeInt
    frame_timestamp: int
    timestamp: int
    champion_kill_event_id: str
    direction: Literal["DEALT", "RECEIVED"]
    idx: NonNegativeInt


class ChampionSpecialKillRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["CHAMPION_SPECIAL_KILL"]
    timestamp: int
    killType: str
    killerId: int
    position_x: int
    position_y: int
    multiKillLength: int


class DragonSoulGivenRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["DRAGON_SOUL_GIVEN"]
    timestamp: int
    name: str
    teamId: int


class EliteMonsterKillRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["ELITE_MONSTER_KILL"]
    timestamp: int
    assistingParticipantIds: list[int]
    bounty: int
    killerId: int
    killerTeamId: int
    monsterSubType: str
    monsterType: str
    position_x: int
    position_y: int


class GameEndRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["GAME_END"]
    timestamp: int
    gameId: int
    realTimestamp: int
    winningTeam: int


class ItemDestroyedRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["ITEM_DESTROYED"]
    timestamp: int
    itemId: int
    participantId: int


class ItemPurchasedRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["ITEM_PURCHASED"]
    timestamp: int
    itemId: int
    participantId: int


class ItemSoldRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["ITEM_SOLD"]
    timestamp: int
    itemId: int
    participantId: int


class ItemUndoRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["ITEM_UNDO"]
    timestamp: int
    afterId: int
    beforeId: int
    goldGain: int
    participantId: int


class LevelUpRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["LEVEL_UP"]
    timestamp: int
    level: int
    participantId: int


class PauseEndRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["PAUSE_END"]
    timestamp: int
    realTimestamp: int


class SkillLevelUpRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["SKILL_LEVEL_UP"]
    timestamp: int
    levelUpType: str
    participantId: int
    skillSlot: int


class TurretPlateDestroyedRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["TURRET_PLATE_DESTROYED"]
    timestamp: int
    killerId: int
    laneType: str
    position_x: int
    position_y: int
    teamId: int


class WardKillRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["WARD_KILL"]
    timestamp: int
    killerId: int
    wardType: str


class WardPlacedRow(TypedDict):
    gameId: int
    frame_timestamp: NonNegativeInt
    type: Literal["WARD_PLACED"]
    timestamp: int
    creatorId: int
    wardType: str


RowT = TypeVar("RowT", bound=EventBase)


class EventTypeParser(Generic[RowT]):
    EVENT_TYPE: ClassVar[str]

    def parse(self, frames: list[Frame], gameId: int) -> list[RowT]:
        rows: list[RowT] = []

        for frame in frames:
            frame_timestamp = frame.timestamp

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


class BuildingKillParser(EventTypeParser[BuildingKillRow]):
    EVENT_TYPE = "BUILDING_KILL"


class ChampionKillParser(EventTypeParser[ChampionKillRow]):
    EVENT_TYPE = "CHAMPION_KILL"

    def parse(self, frames: list[Frame], gameId: int) -> list[ChampionKillRow]:
        rows: list[ChampionKillRow] = []

        for frame in frames:
            frame_ts = frame.timestamp

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
            frame_ts = int(frame.timestamp)

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
    participantStats: list[ParticipantStatsRow]

    buildingKill: list[BuildingKillRow]
    championKill: list[ChampionKillRow]
    championSpecialKill: list[ChampionSpecialKillRow]
    dragonSoulGiven: list[DragonSoulGivenRow]
    eliteMonsterKill: list[EliteMonsterKillRow]
    gameEnd: list[GameEndRow]

    itemDestroyed: list[ItemDestroyedRow]
    itemPurchased: list[ItemPurchasedRow]
    itemSold: list[ItemSoldRow]
    itemUndo: list[ItemUndoRow]

    levelUp: list[LevelUpRow]
    pauseEnd: list[PauseEndRow]
    skillLevelUp: list[SkillLevelUpRow]

    turretPlateDestroyed: list[TurretPlateDestroyedRow]
    wardKill: list[WardKillRow]
    wardPlaced: list[WardPlacedRow]

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
    gameEnd: EventParser[list[Frame], list[GameEndRow]] = field(
        default_factory=GameEndParser
    )

    itemDestroyed: EventParser[list[Frame], list[ItemDestroyedRow]] = field(
        default_factory=ItemDestroyedParser
    )
    itemPurchased: EventParser[list[Frame], list[ItemPurchasedRow]] = field(
        default_factory=ItemPurchasedParser
    )
    itemSold: EventParser[list[Frame], list[ItemSoldRow]] = field(
        default_factory=ItemSoldParser
    )
    itemUndo: EventParser[list[Frame], list[ItemUndoRow]] = field(
        default_factory=ItemUndoParser
    )

    levelUp: EventParser[list[Frame], list[LevelUpRow]] = field(
        default_factory=LevelUpParser
    )
    pauseEnd: EventParser[list[Frame], list[PauseEndRow]] = field(
        default_factory=PauseEndParser
    )
    skillLevelUp: EventParser[list[Frame], list[SkillLevelUpRow]] = field(
        default_factory=SkillLevelUpParser
    )

    turretPlateDestroyed: EventParser[list[Frame], list[TurretPlateDestroyedRow]] = (
        field(default_factory=TurretPlateDestroyedParser)
    )
    wardKill: EventParser[list[Frame], list[WardKillRow]] = field(
        default_factory=WardKillParser
    )
    wardPlaced: EventParser[list[Frame], list[WardPlacedRow]] = field(
        default_factory=WardPlacedParser
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
            gameEnd=self.gameEnd.parse(frames, gameId),
            itemDestroyed=self.itemDestroyed.parse(frames, gameId),
            itemPurchased=self.itemPurchased.parse(frames, gameId),
            itemSold=self.itemSold.parse(frames, gameId),
            itemUndo=self.itemUndo.parse(frames, gameId),
            levelUp=self.levelUp.parse(frames, gameId),
            pauseEnd=self.pauseEnd.parse(frames, gameId),
            skillLevelUp=self.skillLevelUp.parse(frames, gameId),
            turretPlateDestroyed=self.turretPlateDestroyed.parse(frames, gameId),
            wardKill=self.wardKill.parse(frames, gameId),
            wardPlaced=self.wardPlaced.parse(frames, gameId),
            championKillVictimDamageDealt=self.championKillVictimDamageDealt.parse(
                frames, gameId
            ),
            championKillVictimDamageReceived=self.championKillVictimDamageReceived.parse(
                frames, gameId
            ),
        )
