from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import (
    Any,
    ClassVar,
    Literal,
    TypedDict,
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

FRAME_TIMESTAMP_BUCKET_MS = 60_000


def nearest_frame_timestamp(timestamp_ms: int) -> int:
    ts = int(timestamp_ms)
    if ts <= 0:
        return 0
    return ((ts + (FRAME_TIMESTAMP_BUCKET_MS // 2)) // FRAME_TIMESTAMP_BUCKET_MS) * (
        FRAME_TIMESTAMP_BUCKET_MS
    )


def champion_kill_event_id(
    *,
    matchId: str | int,
    timestamp: int,
    killerId: int,
    victimId: int,
) -> str:
    return f"{matchId}:{timestamp}:{killerId}:{victimId}"


def _flatten_position(row: dict[str, Any]) -> None:
    """Expand an event ``position`` (dict or Position model) into
    ``position_x`` / ``position_y`` and drop the original key."""
    pos = row.get("position")
    if pos is None:
        return
    if isinstance(pos, dict):
        row["position_x"] = pos["x"]
        row["position_y"] = pos["y"]
    else:
        row["position_x"] = pos.x
        row["position_y"] = pos.y
    row.pop("position", None)


class MatchIdentityRow(TypedDict):
    matchId: str | NonNegativeInt


class ParticipantStatsRow(MatchIdentityRow):
    frame_timestamp: NonNegativeInt
    participantId: NonNegativeInt

    abilityHaste: int
    abilityPower: int
    armor: int
    armorPen: int
    armorPenPercent: int
    attackDamage: int
    attackSpeed: int
    bonusArmorPenPercent: int
    bonusMagicPenPercent: int
    ccReduction: int
    cooldownReduction: int
    health: int
    healthMax: int
    healthRegen: int
    lifesteal: int
    magicPen: int
    magicPenPercent: int
    magicResist: int
    movementSpeed: int
    omnivamp: int
    physicalVamp: int
    power: int
    powerMax: int
    powerRegen: int
    spellVamp: int

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
    def parse(self, frames: list[Frame], matchId: str | int) -> list[ParticipantStatsRow]:
        rows: list[ParticipantStatsRow] = []

        for frame in frames:
            frame_timestamp: NonNegativeInt = cast(
                NonNegativeInt, nearest_frame_timestamp(frame.timestamp)
            )

            for pf in frame.participantFrames.root.values():
                row_dict: dict[str, Any] = {
                    "matchId": matchId,
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


class TimelineEventRowBase(MatchIdentityRow):
    frame_timestamp: NonNegativeInt
    timestamp: int


class BuildingKillRow(TimelineEventRowBase):
    type: Literal["BUILDING_KILL"]
    bounty: NonNegativeInt
    buildingType: str
    assistingParticipantIds: list[int]
    killerId: int
    laneType: str
    position_x: int
    position_y: int
    teamId: NonNegativeInt
    towerType: str | None


class ChampionKillRow(TimelineEventRowBase):
    type: Literal["CHAMPION_KILL"]
    champion_kill_event_id: str
    assistingParticipantIds: list[int]
    killerId: int
    victimId: int
    bounty: int
    killStreakLength: int
    shutdownBounty: int
    position_x: int
    position_y: int


class ChampionKillDamageInstanceRow(DamageInstance, MatchIdentityRow):
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
    assistingParticipantIds: list[int]
    bounty: int
    killerId: int
    killerTeamId: int
    monsterSubType: str | None
    monsterType: str
    position_x: int
    position_y: int


class WardPlacedRow(TimelineEventRowBase):
    creatorId: NonNegativeInt
    wardType: str


class WardKillRow(TimelineEventRowBase):
    killerId: int
    wardType: str


class ItemPurchasedRow(TimelineEventRowBase):
    participantId: int
    itemId: int


class ItemSoldRow(TimelineEventRowBase):
    participantId: int
    itemId: int


class ItemDestroyedRow(TimelineEventRowBase):
    participantId: int
    itemId: int


class ItemUndoRow(TimelineEventRowBase):
    participantId: int
    beforeId: int
    afterId: int
    goldGain: int


class LevelUpRow(TimelineEventRowBase):
    participantId: int
    level: int


class SkillLevelUpRow(TimelineEventRowBase):
    participantId: int
    skillSlot: int
    levelUpType: str


class PauseEndRow(TimelineEventRowBase):
    realTimestamp: int


class GameEndRow(TimelineEventRowBase):
    winningTeam: int
    gameId: int | None
    realTimestamp: int


class ObjectiveBountyPrestartRow(TimelineEventRowBase):
    teamId: int
    actualStartTime: int


class ObjectiveBountyFinishRow(TimelineEventRowBase):
    teamId: int


class FeatUpdateRow(TimelineEventRowBase):
    teamId: int
    featType: int
    featValue: int


class ChampionTransformRow(TimelineEventRowBase):
    participantId: int
    transformType: str


class TurretPlateDestroyedRow(TimelineEventRowBase):
    type: Literal["TURRET_PLATE_DESTROYED"]
    killerId: int
    laneType: str
    position_x: int
    position_y: int
    teamId: int


class EventTypeParser[RowT]:
    EVENT_TYPE: ClassVar[str]
    INCLUDE_TYPE: ClassVar[bool] = False
    # Keys defaulted via setdefault when the raw event omits them.
    DEFAULTS: ClassVar[dict[str, Any]] = {}
    # Keys normalised to [] when missing or falsy.
    EMPTY_LIST_FIELDS: ClassVar[tuple[str, ...]] = ()

    def _build_row(
        self, e: dict[str, Any], frame_timestamp: int, matchId: str | int
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            **e,
            "frame_timestamp": frame_timestamp,
            "matchId": matchId,
        }
        if not self.INCLUDE_TYPE:
            row.pop("type", None)
        for key, default in self.DEFAULTS.items():
            row.setdefault(key, default)
        for key in self.EMPTY_LIST_FIELDS:
            if not row.get(key):
                row[key] = []
        _flatten_position(row)
        return row

    def parse(self, frames: list[Frame], matchId: str | int) -> list[RowT]:
        rows: list[RowT] = []

        for frame in frames:
            frame_timestamp = nearest_frame_timestamp(frame.timestamp)

            for e in frame.events:
                if e["type"] != self.EVENT_TYPE:
                    continue

                rows.append(cast(RowT, self._build_row(e, frame_timestamp, matchId)))

        return rows


class ChampionKillParser(EventTypeParser[ChampionKillRow]):
    EVENT_TYPE = "CHAMPION_KILL"

    def parse(self, frames: list[Frame], matchId: str | int) -> list[ChampionKillRow]:
        rows: list[ChampionKillRow] = []

        for frame in frames:
            frame_ts = nearest_frame_timestamp(frame.timestamp)

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
                assisting_ids = e2.get("assistingParticipantIds")
                row["assistingParticipantIds"] = assisting_ids if assisting_ids else []

                _flatten_position(row)

                rows.append(cast(ChampionKillRow, row))

        return rows


class ChampionKillDamageInstanceParser:
    KEY: ClassVar[Literal["victimDamageDealt", "victimDamageReceived"]]
    ALIAS_KEY: ClassVar[
        Literal["victimTeamfightDamageDealt", "victimTeamfightDamageReceived"]
    ]
    DIRECTION: ClassVar[Literal["DEALT", "RECEIVED"]]

    def parse(
        self, frames: list[Frame], matchId: str | int
    ) -> list[ChampionKillDamageInstanceRow]:
        rows: list[ChampionKillDamageInstanceRow] = []

        for frame in frames:
            frame_ts = nearest_frame_timestamp(frame.timestamp)

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
    INCLUDE_TYPE = True
    DEFAULTS = {"multiKillLength": None}


class DragonSoulGivenParser(EventTypeParser[DragonSoulGivenRow]):
    EVENT_TYPE = "DRAGON_SOUL_GIVEN"
    INCLUDE_TYPE = True


class EliteMonsterKillParser(EventTypeParser[EliteMonsterKillRow]):
    EVENT_TYPE = "ELITE_MONSTER_KILL"
    INCLUDE_TYPE = True
    DEFAULTS = {"monsterSubType": None}
    EMPTY_LIST_FIELDS = ("assistingParticipantIds",)


class WardPlacedParser(EventTypeParser[WardPlacedRow]):
    EVENT_TYPE = "WARD_PLACED"


class WardKillParser(EventTypeParser[WardKillRow]):
    EVENT_TYPE = "WARD_KILL"


class ItemPurchasedParser(EventTypeParser[ItemPurchasedRow]):
    EVENT_TYPE = "ITEM_PURCHASED"


class ItemSoldParser(EventTypeParser[ItemSoldRow]):
    EVENT_TYPE = "ITEM_SOLD"


class ItemDestroyedParser(EventTypeParser[ItemDestroyedRow]):
    EVENT_TYPE = "ITEM_DESTROYED"


class ItemUndoParser(EventTypeParser[ItemUndoRow]):
    EVENT_TYPE = "ITEM_UNDO"


class LevelUpParser(EventTypeParser[LevelUpRow]):
    EVENT_TYPE = "LEVEL_UP"


class SkillLevelUpParser(EventTypeParser[SkillLevelUpRow]):
    EVENT_TYPE = "SKILL_LEVEL_UP"


class PauseEndParser(EventTypeParser[PauseEndRow]):
    EVENT_TYPE = "PAUSE_END"


class GameEndParser(EventTypeParser[GameEndRow]):
    EVENT_TYPE = "GAME_END"
    DEFAULTS = {"gameId": None}


class ObjectiveBountyPrestartParser(EventTypeParser[ObjectiveBountyPrestartRow]):
    EVENT_TYPE = "OBJECTIVE_BOUNTY_PRESTART"


class ObjectiveBountyFinishParser(EventTypeParser[ObjectiveBountyFinishRow]):
    EVENT_TYPE = "OBJECTIVE_BOUNTY_FINISH"


class FeatUpdateParser(EventTypeParser[FeatUpdateRow]):
    EVENT_TYPE = "FEAT_UPDATE"


class ChampionTransformParser(EventTypeParser[ChampionTransformRow]):
    EVENT_TYPE = "CHAMPION_TRANSFORM"


class TurretPlateDestroyedParser(EventTypeParser[TurretPlateDestroyedRow]):
    EVENT_TYPE = "TURRET_PLATE_DESTROYED"
    INCLUDE_TYPE = True


class BuildingKillParser(EventTypeParser[BuildingKillRow]):
    EVENT_TYPE = "BUILDING_KILL"
    INCLUDE_TYPE = True
    DEFAULTS = {"towerType": None}
    EMPTY_LIST_FIELDS = ("assistingParticipantIds",)


@dataclass
class TimelineTables:
    participantStats: list[ParticipantStatsRow]

    buildingKill: list[BuildingKillRow]
    championKill: list[ChampionKillRow]
    championSpecialKill: list[ChampionSpecialKillRow]
    dragonSoulGiven: list[DragonSoulGivenRow]
    eliteMonsterKill: list[EliteMonsterKillRow]

    wardPlaced: list[WardPlacedRow]
    wardKill: list[WardKillRow]
    itemPurchased: list[ItemPurchasedRow]
    itemSold: list[ItemSoldRow]
    itemDestroyed: list[ItemDestroyedRow]
    itemUndo: list[ItemUndoRow]
    levelUp: list[LevelUpRow]
    skillLevelUp: list[SkillLevelUpRow]
    pauseEnd: list[PauseEndRow]
    gameEnd: list[GameEndRow]
    objectiveBountyPrestart: list[ObjectiveBountyPrestartRow]
    objectiveBountyFinish: list[ObjectiveBountyFinishRow]
    featUpdate: list[FeatUpdateRow]
    championTransform: list[ChampionTransformRow]

    turretPlateDestroyed: list[TurretPlateDestroyedRow]

    championKillVictimDamageDealt: list[ChampionKillDamageInstanceRow]
    championKillVictimDamageReceived: list[ChampionKillDamageInstanceRow]

    @classmethod
    def empty(cls) -> TimelineTables:
        return cls(**{f.name: [] for f in fields(cls)})


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

    wardPlaced: EventParser[list[Frame], list[WardPlacedRow]] = field(
        default_factory=WardPlacedParser
    )
    wardKill: EventParser[list[Frame], list[WardKillRow]] = field(
        default_factory=WardKillParser
    )
    itemPurchased: EventParser[list[Frame], list[ItemPurchasedRow]] = field(
        default_factory=ItemPurchasedParser
    )
    itemSold: EventParser[list[Frame], list[ItemSoldRow]] = field(
        default_factory=ItemSoldParser
    )
    itemDestroyed: EventParser[list[Frame], list[ItemDestroyedRow]] = field(
        default_factory=ItemDestroyedParser
    )
    itemUndo: EventParser[list[Frame], list[ItemUndoRow]] = field(
        default_factory=ItemUndoParser
    )
    levelUp: EventParser[list[Frame], list[LevelUpRow]] = field(
        default_factory=LevelUpParser
    )
    skillLevelUp: EventParser[list[Frame], list[SkillLevelUpRow]] = field(
        default_factory=SkillLevelUpParser
    )
    pauseEnd: EventParser[list[Frame], list[PauseEndRow]] = field(
        default_factory=PauseEndParser
    )
    gameEnd: EventParser[list[Frame], list[GameEndRow]] = field(
        default_factory=GameEndParser
    )
    objectiveBountyPrestart: EventParser[
        list[Frame], list[ObjectiveBountyPrestartRow]
    ] = field(default_factory=ObjectiveBountyPrestartParser)
    objectiveBountyFinish: EventParser[
        list[Frame], list[ObjectiveBountyFinishRow]
    ] = field(default_factory=ObjectiveBountyFinishParser)
    featUpdate: EventParser[list[Frame], list[FeatUpdateRow]] = field(
        default_factory=FeatUpdateParser
    )
    championTransform: EventParser[list[Frame], list[ChampionTransformRow]] = field(
        default_factory=ChampionTransformParser
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

    @staticmethod
    def _is_abort_unexpected_payload(raw: dict[str, Any]) -> bool:
        info = raw.get("info")
        if not isinstance(info, dict):
            return False

        end_of_game_result = info.get("endOfGameResult")
        frames = info.get("frames")
        first_frame = frames[0] if isinstance(frames, list) and frames else None

        return (
            isinstance(end_of_game_result, str)
            and end_of_game_result.startswith("Abort")
        ) or (
            isinstance(first_frame, dict)
            and first_frame.get("participantFrames") is None
        )

    @staticmethod
    def _is_unsupported_game_mode(raw: dict[str, Any]) -> bool:
        # Timeline payloads omit gameMode, so detect SWARM (and any other small-team
        # PvE mode) by participant count: every supported standard/Arena mode has
        # >=5 puuids in metadata.participants; SWARM is 1-4.
        metadata = raw.get("metadata")
        participants = metadata.get("participants") if isinstance(metadata, dict) else None
        return isinstance(participants, list) and 0 < len(participants) < 5

    def run(self, raw: dict[str, Any]) -> TimelineTables:
        metadata_raw = raw.get("metadata", {})
        match_id = (
            metadata_raw.get("matchId", "unknown")
            if isinstance(metadata_raw, dict)
            else "unknown"
        )

        if self._is_unsupported_game_mode(raw):
            logger.info(
                "TimelineSkip match_id=%s reason=unsupported_game_mode (participant_count<5)",
                match_id,
            )
            return TimelineTables.empty()

        drift_date = self._drift_date()
        timeline_drift(raw, match_id=match_id, drift_date=drift_date)

        if self._is_abort_unexpected_payload(raw):
            info = raw.get("info")
            end_of_game_result = (
                info.get("endOfGameResult") if isinstance(info, dict) else "unknown"
            )
            logger.warning(
                "TimelineAbort match_id=%s date=%s endOfGameResult=%s; emitting empty timeline tables.",
                match_id,
                drift_date,
                end_of_game_result,
            )
            return TimelineTables.empty()

        try:
            tl = Timeline.model_validate(raw)
            metadata = tl.metadata
            info = tl.info
            frames = info.frames
            matchId = metadata.matchId

            tables = TimelineTables(
                participantStats=self.participantStats.parse(frames, matchId),
                buildingKill=self.buildingKill.parse(frames, matchId),
                championKill=self.championKill.parse(frames, matchId),
                championSpecialKill=self.championSpecialKill.parse(frames, matchId),
                dragonSoulGiven=self.dragonSoulGiven.parse(frames, matchId),
                eliteMonsterKill=self.eliteMonsterKill.parse(frames, matchId),
                wardPlaced=self.wardPlaced.parse(frames, matchId),
                wardKill=self.wardKill.parse(frames, matchId),
                itemPurchased=self.itemPurchased.parse(frames, matchId),
                itemSold=self.itemSold.parse(frames, matchId),
                itemDestroyed=self.itemDestroyed.parse(frames, matchId),
                itemUndo=self.itemUndo.parse(frames, matchId),
                levelUp=self.levelUp.parse(frames, matchId),
                skillLevelUp=self.skillLevelUp.parse(frames, matchId),
                pauseEnd=self.pauseEnd.parse(frames, matchId),
                gameEnd=self.gameEnd.parse(frames, matchId),
                objectiveBountyPrestart=self.objectiveBountyPrestart.parse(
                    frames, matchId
                ),
                objectiveBountyFinish=self.objectiveBountyFinish.parse(
                    frames, matchId
                ),
                featUpdate=self.featUpdate.parse(frames, matchId),
                championTransform=self.championTransform.parse(frames, matchId),
                turretPlateDestroyed=self.turretPlateDestroyed.parse(frames, matchId),
                championKillVictimDamageDealt=self.championKillVictimDamageDealt.parse(
                    frames, matchId
                ),
                championKillVictimDamageReceived=self.championKillVictimDamageReceived.parse(
                    frames, matchId
                ),
            )
        except ValidationError as e:
            errs = e.errors(include_input=True)
            logger.warning(
                "SchemaValidation timeline match_id=%s date=%s errors=%s",
                match_id,
                drift_date,
                e.errors(include_input=False),
            )
            logger.warning(
                "SchemaValidation timeline value=%r",
                errs[-1].get("input") if errs else None,
            )
            logger.warning(
                "Aborting timeline payload for match_id=%s due to validation errors.",
                match_id,
            )
            raise ValueError(
                f"Schema validation failed for timeline payload match_id={match_id}"
            ) from e
        return tables
