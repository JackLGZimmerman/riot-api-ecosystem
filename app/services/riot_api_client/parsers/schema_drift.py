from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.riot_api_client.parsers.models import timeline as tl_models
from app.services.riot_api_client.parsers.models.non_timeline import (
    Ban,
    Challenges,
    Feats,
    Info,
    Metadata,
    Objectives,
    Participant,
    Perks,
)


class SchemaDriftError(RuntimeError):
    pass


DRIFT_OUTPUT_DIR = Path("app/core/logging/logs/schema_drift")

NON_TIMELINE_CHECKS: dict[str, dict[str, Any]] = {
    "metadata": {"model": Metadata, "path": ["metadata"]},
    "info": {"model": Info, "path": ["info"]},
    "bans": {"model": Ban, "path": ["info", "teams", "*", "bans", "*"]},
    "feats": {
        "model": Feats,
        "path": ["info", "teams", "*", "feats"],
        "optional": True,
    },
    "objectives": {"model": Objectives, "path": ["info", "teams", "*", "objectives"]},
    "participants": {"model": Participant, "path": ["info", "participants", "*"]},
    "challenges": {
        "model": Challenges,
        "path": ["info", "participants", "*", "challenges"],
    },
    "perks": {"model": Perks, "path": ["info", "participants", "*", "perks"]},
}

TIMELINE_EVENTS: dict[str, Any] = {
    "BUILDING_KILL": tl_models.EventBuildingKill,
    "CHAMPION_KILL": tl_models.EventChampionKill,
    "CHAMPION_SPECIAL_KILL": tl_models.EventChampionSpecialKill,
    "CHAMPION_TRANSFORM": tl_models.EventChampionTransform,
    "DRAGON_SOUL_GIVEN": tl_models.EventDragonSoulGiven,
    "ELITE_MONSTER_KILL": tl_models.EventEliteMonsterKill,
    "FEAT_UPDATE": tl_models.EventFeatUpdate,
    "GAME_END": tl_models.EventGameEnd,
    "ITEM_DESTROYED": tl_models.EventItemDestroyed,
    "ITEM_PURCHASED": tl_models.EventItemPurchased,
    "ITEM_SOLD": tl_models.EventItemSold,
    "ITEM_UNDO": tl_models.EventItemUndo,
    "LEVEL_UP": tl_models.EventLevelUp,
    "OBJECTIVE_BOUNTY_FINISH": tl_models.EventObjectiveBountyFinish,
    "OBJECTIVE_BOUNTY_PRESTART": tl_models.EventObjectiveBountyPrestart,
    "PAUSE_END": tl_models.EventPauseEnd,
    "SKILL_LEVEL_UP": tl_models.EventSkillLevelUp,
    "TURRET_PLATE_DESTROYED": tl_models.EventTurretPlateDestroyed,
    "UNKNOWN": tl_models.UnknownEvent,
    "WARD_KILL": tl_models.EventWardKill,
    "WARD_PLACED": tl_models.EventWardPlaced,
}


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: type(item).__name__ for key, item in sorted(value.items())}
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "item_type": type(value[0]).__name__ if value else None,
        }
    return type(value).__name__


def _model_schema(model: Any) -> tuple[dict[str, Any], set[str], set[str]]:
    required: set[str] = set()
    optional: set[str] = set()
    types: dict[str, str] = {}
    for name, field in model.model_fields.items():
        keys = [name, field.alias] if field.alias else [name]
        target = required if field.is_required() else optional
        for key in keys:
            target.add(key)
            types[key] = str(field.annotation)
    return _schema(required, optional, types), required | optional, required


def _typed_dict_schema(model: Any) -> tuple[dict[str, Any], set[str], set[str]]:
    required = set(getattr(model, "__required_keys__", set()))
    optional = set(getattr(model, "__optional_keys__", set()))
    annotations = getattr(model, "__annotations__", {})
    types = {key: str(annotations.get(key, "unknown")) for key in required | optional}
    return _schema(required, optional, types), required | optional, required


def _schema(required: set[str], optional: set[str], types: dict[str, str]) -> dict[str, Any]:
    return {
        "required": sorted(required),
        "optional": sorted(optional),
        "types": {key: types[key] for key in sorted(types)},
    }


def _diff(expected: set[str], required: set[str], actual: dict[str, Any]) -> list[Any]:
    actual_keys = set(actual)
    differences: list[Any] = []
    missing = sorted(required - actual_keys)
    unexpected = sorted(actual_keys - expected)
    if missing:
        differences.append({"type": "missing_keys", "keys": missing})
    if unexpected:
        differences.append(
            {
                "type": "unexpected_keys",
                "keys": unexpected,
                "examples": {key: actual[key] for key in unexpected[:10]},
            }
        )
    return differences


def _resolve(raw: Any, path: list[str]) -> tuple[list[tuple[str, Any]], dict[str, Any] | None]:
    nodes = [("$", raw)]
    for token in path:
        next_nodes = []
        for node_path, node in nodes:
            if token == "*":
                if not isinstance(node, list):
                    return [], {
                        "type": "expected_list",
                        "path": node_path,
                        "actual_schema": _shape(node),
                    }
                next_nodes.extend((f"{node_path}[{idx}]", item) for idx, item in enumerate(node))
            elif not isinstance(node, dict):
                return [], {
                    "type": "expected_object",
                    "path": node_path,
                    "actual_schema": _shape(node),
                }
            elif token not in node:
                return [], {
                    "type": "missing_path",
                    "path": node_path,
                    "missing": token,
                    "actual_schema": _shape(node),
                }
            else:
                next_nodes.append((f"{node_path}.{token}", node[token]))
        nodes = next_nodes
    return nodes, None


def _fail(
    *,
    stream: str,
    match_id: str,
    drift_date: str,
    checked_object: str,
    path: str,
    expected_schema: Any,
    actual_schema: Any,
    differences: list[Any],
) -> None:
    report = {
        "detected_at": datetime.now(tz=UTC).isoformat(),
        "stream": stream,
        "match_id": match_id,
        "drift_date": drift_date,
        "object": checked_object,
        "path": path,
        "expected_schema": expected_schema,
        "actual_schema": actual_schema,
        "differences": differences,
    }
    output_dir = Path(os.getenv("SCHEMA_DRIFT_DIR", str(DRIFT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{stream}_{match_id}_{checked_object}")
    report_path = output_dir / f"{datetime.now(tz=UTC):%Y%m%dT%H%M%S%fZ}_{safe_name}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    raise SchemaDriftError(
        "Schema drift detected "
        f"stream={stream} match_id={match_id} object={checked_object} "
        f"path={path} report={report_path}"
    )


def _check(
    *,
    stream: str,
    match_id: str,
    drift_date: str,
    checked_object: str,
    path: str,
    expected_schema: Any,
    expected_keys: set[str],
    required_keys: set[str],
    actual: Any,
) -> None:
    if not isinstance(actual, dict):
        _fail(
            stream=stream,
            match_id=match_id,
            drift_date=drift_date,
            checked_object=checked_object,
            path=path,
            expected_schema=expected_schema,
            actual_schema=_shape(actual),
            differences=[{"type": "expected_object"}],
        )
    differences = _diff(expected_keys, required_keys, actual)
    if differences:
        _fail(
            stream=stream,
            match_id=match_id,
            drift_date=drift_date,
            checked_object=checked_object,
            path=path,
            expected_schema=expected_schema,
            actual_schema=_shape(actual),
            differences=differences,
        )


def non_timeline(raw: Any, *, match_id: str = "unknown", drift_date: str = "unknown") -> None:
    for checked_object, check in NON_TIMELINE_CHECKS.items():
        model = check["model"]
        expected_schema, expected_keys, required_keys = _model_schema(model)
        nodes, error = _resolve(raw, check["path"])
        if error:
            if check.get("optional"):
                continue
            _fail(
                stream="non_timeline",
                match_id=match_id,
                drift_date=drift_date,
                checked_object=checked_object,
                path=error["path"],
                expected_schema=expected_schema,
                actual_schema=error["actual_schema"],
                differences=[error],
            )
        for node_path, node in nodes:
            _check(
                stream="non_timeline",
                match_id=match_id,
                drift_date=drift_date,
                checked_object=checked_object,
                path=node_path,
                expected_schema=expected_schema,
                expected_keys=expected_keys,
                required_keys=required_keys,
                actual=node,
            )


def timeline(raw: Any, *, match_id: str = "unknown", drift_date: str = "unknown") -> None:
    info = raw.get("info") if isinstance(raw, dict) else None
    frames = info.get("frames") if isinstance(info, dict) else None
    if not isinstance(frames, list):
        _fail(
            stream="timeline",
            match_id=match_id,
            drift_date=drift_date,
            checked_object="frames",
            path="$.info.frames",
            expected_schema="list[Frame]",
            actual_schema=_shape(frames),
            differences=[{"type": "expected_list"}],
        )

    for frame_idx, frame in enumerate(frames):
        events = frame.get("events") if isinstance(frame, dict) else None
        if not isinstance(events, list):
            _fail(
                stream="timeline",
                match_id=match_id,
                drift_date=drift_date,
                checked_object="events",
                path=f"$.info.frames[{frame_idx}].events",
                expected_schema="list[Event]",
                actual_schema=_shape(events),
                differences=[{"type": "expected_list"}],
            )

        for event_idx, event in enumerate(events):
            path = f"$.info.frames[{frame_idx}].events[{event_idx}]"
            event_type = event.get("type") if isinstance(event, dict) else None
            model = TIMELINE_EVENTS.get(event_type)
            if model is None:
                _fail(
                    stream="timeline",
                    match_id=match_id,
                    drift_date=drift_date,
                    checked_object="event",
                    path=path,
                    expected_schema=sorted(TIMELINE_EVENTS),
                    actual_schema=_shape(event),
                    differences=[{"type": "unknown_event_type", "event_type": event_type}],
                )

            expected_schema, expected_keys, required_keys = _typed_dict_schema(model)
            _check(
                stream="timeline",
                match_id=match_id,
                drift_date=drift_date,
                checked_object=f"event:{event_type}",
                path=path,
                expected_schema=expected_schema,
                expected_keys=expected_keys,
                required_keys=required_keys,
                actual=event,
            )
