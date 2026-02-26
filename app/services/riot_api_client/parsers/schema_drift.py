from __future__ import annotations

import logging
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

SchemaValidation = dict[str, Any]
drift_logger = logging.getLogger("schema_drift.non_timeline")
timeline_drift_logger = logging.getLogger("schema_drift.timeline")

NON_TIMELINE_SCHEMA = {
    "metadata": {"structure": Metadata, "path": ["metadata"]},
    "info": {"structure": Info, "path": ["info"]},
    "bans": {"structure": Ban, "path": ["info", "teams", "*", "bans", "*"]},
    "feats": {
        "structure": Feats,
        "path": ["info", "teams", "*", "feats"],
        "optional_path": True,
    },
    "objectives": {
        "structure": Objectives,
        "path": ["info", "teams", "*", "objectives"],
    },
    "participants": {"structure": Participant, "path": ["info", "participants", "*"]},
    "challenges": {
        "structure": Challenges,
        "path": ["info", "participants", "*", "challenges"],
    },
    "perks": {"structure": Perks, "path": ["info", "participants", "*", "perks"]},
}


def _resolve_path(
    raw: Any, path: list[str]
) -> tuple[list[tuple[str, Any]], SchemaValidation | None]:
    nodes: list[tuple[str, Any]] = [("$", raw)]

    for token in path:
        next_nodes: list[tuple[str, Any]] = []
        for node_path, node in nodes:
            if token == "*":
                if not isinstance(node, list):
                    return [], {
                        "path": node_path,
                        "error_type": "expected_list_for_wildcard",
                        "message": f"Expected list at '{node_path}' for wildcard '*'.",
                    }

                for idx, item in enumerate(node):
                    next_nodes.append((f"{node_path}[{idx}]", item))
                continue

            if not isinstance(node, dict):
                return [], {
                    "path": node_path,
                    "error_type": "expected_object_for_field",
                    "message": f"Expected object at '{node_path}' before reading field '{token}'.",
                }

            if token not in node:
                return [], {
                    "path": node_path,
                    "error_type": "missing_path_segment",
                    "message": f"Missing expected field '{token}' while resolving '{'.'.join(path)}'.",
                }

            child_path = f"{node_path}.{token}" if node_path else token
            next_nodes.append((child_path, node[token]))

        nodes = next_nodes

    return nodes, None


def _expected_model_keys(model: Any) -> set[str]:
    keys: set[str] = set()
    for field_name, field in model.model_fields.items():
        keys.add(field_name)
        if field.alias:
            keys.add(field.alias)
    return keys


def _typed_dict_keys(typed_dict_cls: Any) -> tuple[set[str], set[str]]:
    required = set(getattr(typed_dict_cls, "__required_keys__", set()))
    optional = set(getattr(typed_dict_cls, "__optional_keys__", set()))
    all_keys = required | optional
    return all_keys, required


def non_timeline(
    raw: Any,
    *,
    match_id: str = "unknown",
    drift_date: str = "unknown",
) -> None:
    structure_changes: dict[str, list[SchemaValidation]] = {}

    for schema_key, spec in NON_TIMELINE_SCHEMA.items():
        model = spec["structure"]
        path = spec["path"]
        resolved_nodes, path_issue = _resolve_path(raw, path)

        if path_issue:
            if spec.get("optional_path"):
                continue
            structure_changes[schema_key] = [
                {
                    "schema_key": schema_key,
                    "model": model.__name__,
                    **path_issue,
                }
            ]
            continue

        expected_keys = _expected_model_keys(model)
        first_seen_unknown: dict[str, SchemaValidation] = {}

        for node_path, node in resolved_nodes:
            if not isinstance(node, dict):
                if "__node_not_object__" not in first_seen_unknown:
                    first_seen_unknown["__node_not_object__"] = {
                        "schema_key": schema_key,
                        "model": model.__name__,
                        "path": node_path,
                        "error_type": "node_not_object",
                        "message": f"Resolved node at '{node_path}' is not an object.",
                        "example_structure": node,
                    }
                continue

            for key in sorted(set(node) - expected_keys):
                if key in first_seen_unknown:
                    continue
                first_seen_unknown[key] = {
                    "schema_key": schema_key,
                    "model": model.__name__,
                    "path": f"{node_path}.{key}",
                    "error_type": "unexpected_key",
                    "message": (
                        f"Unexpected key '{key}' at '{node_path}' "
                        f"(raw_keys - model_keys)."
                    ),
                    "new_key": key,
                    "example_structure": node[key],
                }

        if first_seen_unknown:
            structure_changes[schema_key] = list(first_seen_unknown.values())

    messages = [
        f"{issue['schema_key']}:{issue['path']} - {issue['message']}"
        for issues in structure_changes.values()
        for issue in issues
    ]

    if not messages:
        return

    drift_logger.warning(
        "SchemaDrift non_timeline",
        extra={
            "match_id": match_id,
            "keys": messages,
            "structure_changes": structure_changes,
            "structure_drift_count": len(messages),
            "drift_date": drift_date,
        },
    )
    for handler in drift_logger.handlers:
        handler.flush()


def timeline(
    raw: Any, *, match_id: str = "unknown", drift_date: str = "unknown"
) -> None:
    structure_changes: dict[str, list[SchemaValidation]] = {}

    info = raw.get("info") if isinstance(raw, dict) else None
    frames = info.get("frames") if isinstance(info, dict) else None
    if not isinstance(frames, list):
        structure_changes["events"] = [
            {
                "schema_key": "events",
                "model": "Frame.events",
                "path": "$.info.frames",
                "error_type": "missing_or_invalid_frames",
                "message": "Expected '$.info.frames' to be a list.",
                "example_structure": frames,
            }
        ]
    else:
        known_event_models: dict[str, Any] = {
            "ITEM_PURCHASED": tl_models.EventItemPurchased,
            "ITEM_UNDO": tl_models.EventItemUndo,
            "SKILL_LEVEL_UP": tl_models.EventSkillLevelUp,
            "WARD_KILL": tl_models.EventWardKill,
            "WARD_PLACED": tl_models.EventWardPlaced,
            "LEVEL_UP": tl_models.EventLevelUp,
            "GAME_END": tl_models.EventGameEnd,
            "ITEM_DESTROYED": tl_models.EventItemDestroyed,
            "ITEM_SOLD": tl_models.EventItemSold,
            "PAUSE_END": tl_models.EventPauseEnd,
            "CHAMPION_KILL": tl_models.EventChampionKill,
            "CHAMPION_SPECIAL_KILL": tl_models.EventChampionSpecialKill,
            "DRAGON_SOUL_GIVEN": tl_models.EventDragonSoulGiven,
            "ELITE_MONSTER_KILL": tl_models.EventEliteMonsterKill,
            "TURRET_PLATE_DESTROYED": tl_models.EventTurretPlateDestroyed,
            "BUILDING_KILL": tl_models.EventBuildingKill,
            "OBJECTIVE_BOUNTY_PRESTART": tl_models.EventObjectiveBountyPrestart,
            "OBJECTIVE_BOUNTY_FINISH": tl_models.EventObjectiveBountyFinish,
            "FEAT_UPDATE": tl_models.EventFeatUpdate,
            "CHAMPION_TRANSFORM": tl_models.EventChampionTransform,
            "UNKNOWN": tl_models.UnknownEvent,
        }

        first_seen_issues: dict[str, SchemaValidation] = {}
        for frame_idx, frame in enumerate(frames):
            frame_path = f"$.info.frames[{frame_idx}]"
            if not isinstance(frame, dict):
                issue_key = "frame_not_object"
                if issue_key not in first_seen_issues:
                    first_seen_issues[issue_key] = {
                        "schema_key": "events",
                        "model": "Frame",
                        "path": frame_path,
                        "error_type": "frame_not_object",
                        "message": f"Frame at '{frame_path}' is not an object.",
                        "example_structure": frame,
                    }
                continue

            events = frame.get("events")
            if not isinstance(events, list):
                issue_key = "events_not_list"
                if issue_key not in first_seen_issues:
                    first_seen_issues[issue_key] = {
                        "schema_key": "events",
                        "model": "Frame.events",
                        "path": f"{frame_path}.events",
                        "error_type": "events_not_list",
                        "message": f"Expected list at '{frame_path}.events'.",
                        "example_structure": events,
                    }
                continue

            for event_idx, event in enumerate(events):
                event_path = f"{frame_path}.events[{event_idx}]"
                if not isinstance(event, dict):
                    issue_key = "event_not_object"
                    if issue_key not in first_seen_issues:
                        first_seen_issues[issue_key] = {
                            "schema_key": "events",
                            "model": "Event",
                            "path": event_path,
                            "error_type": "event_not_object",
                            "message": f"Event at '{event_path}' is not an object.",
                            "example_structure": event,
                        }
                    continue

                event_type = event.get("type")
                if not isinstance(event_type, str):
                    issue_key = "missing_event_type"
                    if issue_key not in first_seen_issues:
                        first_seen_issues[issue_key] = {
                            "schema_key": "events",
                            "model": "Event",
                            "path": event_path,
                            "error_type": "missing_event_type",
                            "message": f"Event at '{event_path}' is missing string key 'type'.",
                            "example_structure": event,
                        }
                    continue

                event_model = known_event_models.get(event_type)
                if event_model is None:
                    issue_key = f"unknown_event_type:{event_type}"
                    if issue_key not in first_seen_issues:
                        first_seen_issues[issue_key] = {
                            "schema_key": "events",
                            "model": "Event",
                            "path": f"{event_path}.type",
                            "error_type": "unknown_event_type",
                            "message": f"Unknown event type '{event_type}' at '{event_path}'.",
                            "event_type": event_type,
                            "example_structure": event,
                        }
                    continue

                expected_keys, required_keys = _typed_dict_keys(event_model)
                raw_keys = set(event)

                for key in sorted(raw_keys - expected_keys):
                    issue_key = f"{event_type}:unexpected_key:{key}"
                    if issue_key in first_seen_issues:
                        continue
                    first_seen_issues[issue_key] = {
                        "schema_key": "events",
                        "model": event_model.__name__,
                        "path": f"{event_path}.{key}",
                        "error_type": "unexpected_key",
                        "message": (
                            f"Unexpected key '{key}' for event type '{event_type}' "
                            f"(raw_keys - model_keys)."
                        ),
                        "event_type": event_type,
                        "new_key": key,
                        "example_structure": event[key],
                    }

                for key in sorted(required_keys - raw_keys):
                    issue_key = f"{event_type}:missing_required_key:{key}"
                    if issue_key in first_seen_issues:
                        continue
                    first_seen_issues[issue_key] = {
                        "schema_key": "events",
                        "model": event_model.__name__,
                        "path": event_path,
                        "error_type": "missing_required_key",
                        "message": (
                            f"Missing required key '{key}' for event type '{event_type}'."
                        ),
                        "event_type": event_type,
                        "missing_key": key,
                        "example_structure": event,
                    }

        if first_seen_issues:
            structure_changes["events"] = list(first_seen_issues.values())

    messages = [
        f"{issue['schema_key']}:{issue['path']} - {issue['message']}"
        for issues in structure_changes.values()
        for issue in issues
    ]

    if not messages:
        return

    timeline_drift_logger.warning(
        "SchemaDrift timeline",
        extra={
            "match_id": match_id,
            "keys": messages,
            "structure_changes": structure_changes,
            "structure_drift_count": len(messages),
            "drift_date": drift_date,
        },
    )
    for handler in timeline_drift_logger.handlers:
        handler.flush()
