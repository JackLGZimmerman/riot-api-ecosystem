from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Group:
    group_name: str
    parent_group_name: str | None = None
    is_active: bool = True


@dataclass(frozen=True)
class MemberRule:
    member_name: str
    team_position: str | None = None
    champion_id: int | None = None
    build_scope: str | None = None
    is_active: bool = True


@dataclass(frozen=True)
class GroupMember:
    group_name: str
    member_name: str
    is_active: bool = True


@dataclass(frozen=True)
class MetricDefinition:
    metric_name: str
    metric_kind: str
    default_aggregation: str
    is_active: bool = True


@dataclass(frozen=True)
class GroupMetric:
    group_name: str
    metric_name: str
    is_active: bool = True


@dataclass(frozen=True)
class MetricDependency:
    metric_name: str
    depends_on_metric_name: str
    role: str


@dataclass(frozen=True)
class MetadataSnapshot:
    groups: list[Group] = field(default_factory=list)
    members: list[MemberRule] = field(default_factory=list)
    group_members: list[GroupMember] = field(default_factory=list)
    metrics: list[MetricDefinition] = field(default_factory=list)
    group_metrics: list[GroupMetric] = field(default_factory=list)
    metric_dependencies: list[MetricDependency] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedBranch:
    selected_group_name: str
    source_group_name: str
    depth: int
    path: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedMember:
    selected_group_name: str
    source_group_name: str
    matched_member_name: str
    team_position: str | None
    champion_id: int | None
    build_scope: str | None
    specificity: int


@dataclass(frozen=True)
class ResolvedMetric:
    selected_group_name: str
    source_group_name: str
    metric_name: str
    metric_kind: str
    default_aggregation: str
    declared_by_group_name: str


@dataclass(frozen=True)
class DynamicMetricExecutionPlan:
    selected_group_name: str
    branches: list[ResolvedBranch]
    members: list[ResolvedMember]
    metrics: list[ResolvedMetric]
    metric_dependencies: list[MetricDependency]


def load_metadata_snapshot(path: str | Path) -> MetadataSnapshot:
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}

    snapshot = MetadataSnapshot(
        groups=[Group(**row) for row in raw.get("groups", [])],
        members=[MemberRule(**row) for row in raw.get("members", [])],
        group_members=[GroupMember(**row) for row in raw.get("group_members", [])],
        metrics=[MetricDefinition(**row) for row in raw.get("metrics", [])],
        group_metrics=[GroupMetric(**row) for row in raw.get("group_metrics", [])],
        metric_dependencies=[
            MetricDependency(**row) for row in raw.get("metric_dependencies", [])
        ],
    )

    _ensure_unique_names(snapshot.groups, key="group_name", label="group")
    _ensure_unique_names(snapshot.members, key="member_name", label="member")
    _ensure_unique_names(snapshot.metrics, key="metric_name", label="metric")
    return snapshot


def build_execution_plan(
    snapshot: MetadataSnapshot,
    *,
    selected_group_name: str,
) -> DynamicMetricExecutionPlan:
    groups_by_name = {
        group.group_name: group for group in snapshot.groups if group.is_active
    }
    if selected_group_name not in groups_by_name:
        raise KeyError(f"Unknown selected_group_name: {selected_group_name}")

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for group in groups_by_name.values():
        if group.parent_group_name is None:
            continue
        children_by_parent[group.parent_group_name].append(group.group_name)

    members_by_name = {
        member.member_name: member for member in snapshot.members if member.is_active
    }
    group_members_by_group: dict[str, list[str]] = defaultdict(list)
    for group_member in snapshot.group_members:
        if not group_member.is_active:
            continue
        group_members_by_group[group_member.group_name].append(group_member.member_name)

    metrics_by_name = {
        metric.metric_name: metric for metric in snapshot.metrics if metric.is_active
    }
    group_metrics_by_group: dict[str, list[str]] = defaultdict(list)
    for group_metric in snapshot.group_metrics:
        if not group_metric.is_active:
            continue
        group_metrics_by_group[group_metric.group_name].append(group_metric.metric_name)

    branches = _resolve_branches(
        selected_group_name=selected_group_name,
        children_by_parent=children_by_parent,
    )
    members = _resolve_members(
        branches=branches,
        group_members_by_group=group_members_by_group,
        members_by_name=members_by_name,
    )
    metrics = _resolve_metrics(
        branches=branches,
        group_metrics_by_group=group_metrics_by_group,
        metrics_by_name=metrics_by_name,
    )

    active_metric_names = {metric.metric_name for metric in metrics}
    metric_dependencies = [
        dependency
        for dependency in snapshot.metric_dependencies
        if dependency.metric_name in active_metric_names
        and dependency.depends_on_metric_name in active_metric_names
    ]

    return DynamicMetricExecutionPlan(
        selected_group_name=selected_group_name,
        branches=branches,
        members=members,
        metrics=metrics,
        metric_dependencies=metric_dependencies,
    )


def _resolve_branches(
    *,
    selected_group_name: str,
    children_by_parent: dict[str, list[str]],
) -> list[ResolvedBranch]:
    branches: list[ResolvedBranch] = []

    def visit(group_name: str, path: tuple[str, ...]) -> None:
        if group_name in path[:-1]:
            cycle = " -> ".join(path + (group_name,))
            raise ValueError(f"Cycle detected in group definitions: {cycle}")

        branches.append(
            ResolvedBranch(
                selected_group_name=selected_group_name,
                source_group_name=group_name,
                depth=len(path) - 1,
                path=path,
            )
        )

        for child_group_name in sorted(children_by_parent.get(group_name, [])):
            visit(child_group_name, path + (child_group_name,))

    visit(selected_group_name, (selected_group_name,))
    return branches


def _resolve_members(
    *,
    branches: list[ResolvedBranch],
    group_members_by_group: dict[str, list[str]],
    members_by_name: dict[str, MemberRule],
) -> list[ResolvedMember]:
    resolved: list[ResolvedMember] = []

    for branch in branches:
        for group_name in reversed(branch.path):
            member_names = group_members_by_group.get(group_name, [])
            if not member_names:
                continue

            seen_member_names: set[str] = set()
            branch_members: list[ResolvedMember] = []
            for member_name in member_names:
                if member_name in seen_member_names:
                    continue
                seen_member_names.add(member_name)

                member = members_by_name.get(member_name)
                if member is None:
                    raise KeyError(f"Unknown member_name: {member_name}")

                branch_members.append(
                    ResolvedMember(
                        selected_group_name=branch.selected_group_name,
                        source_group_name=branch.source_group_name,
                        matched_member_name=member.member_name,
                        team_position=member.team_position,
                        champion_id=member.champion_id,
                        build_scope=member.build_scope,
                        specificity=_member_specificity(member),
                    )
                )

            branch_members.sort(
                key=lambda member: (-member.specificity, member.matched_member_name)
            )
            resolved.extend(branch_members)
            break

    return resolved


def _resolve_metrics(
    *,
    branches: list[ResolvedBranch],
    group_metrics_by_group: dict[str, list[str]],
    metrics_by_name: dict[str, MetricDefinition],
) -> list[ResolvedMetric]:
    resolved: list[ResolvedMetric] = []

    for branch in branches:
        effective_metrics: dict[str, ResolvedMetric] = {}

        for group_name in branch.path:
            for metric_name in group_metrics_by_group.get(group_name, []):
                metric = metrics_by_name.get(metric_name)
                if metric is None:
                    raise KeyError(f"Unknown metric_name: {metric_name}")

                effective_metrics[metric.metric_name] = ResolvedMetric(
                    selected_group_name=branch.selected_group_name,
                    source_group_name=branch.source_group_name,
                    metric_name=metric.metric_name,
                    metric_kind=metric.metric_kind,
                    default_aggregation=metric.default_aggregation,
                    declared_by_group_name=group_name,
                )

        resolved.extend(effective_metrics.values())

    return resolved


def _member_specificity(member: MemberRule) -> int:
    score = 0

    if member.team_position not in (None, "ANY"):
        score += 1
    if member.champion_id is not None:
        score += 1
    if member.build_scope not in (None, "ANY"):
        score += 1

    return score


def _ensure_unique_names(items: list[Any], *, key: str, label: str) -> None:
    seen: set[str] = set()

    for item in items:
        value = getattr(item, key)
        if value in seen:
            raise ValueError(f"Duplicate {label} name: {value}")
        seen.add(value)
