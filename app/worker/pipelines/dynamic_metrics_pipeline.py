from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Aggregation = Literal["avg", "sum", "min", "max", "count"]
MetricKind = Literal["existing", "derived", "composite"]


@dataclass(frozen=True)
class Group:
    group_id: str
    group_name: str
    parent_group_id: str | None = None
    is_active: bool = True


@dataclass(frozen=True)
class MemberRule:
    member_id: str
    team_position: str | None = None
    champion_id: int | None = None
    build_scope: str | None = None
    is_active: bool = True


@dataclass(frozen=True)
class GroupMember:
    group_id: str
    member_id: str
    is_active: bool = True


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    metric_name: str
    metric_kind: MetricKind
    default_aggregation: Aggregation = "avg"
    is_active: bool = True


@dataclass(frozen=True)
class GroupMetric:
    group_id: str
    metric_id: str
    is_active: bool = True


@dataclass(frozen=True)
class MetricDependency:
    metric_id: str
    depends_on_metric_id: str
    role: str


@dataclass(frozen=True)
class MetadataSnapshot:
    groups: tuple[Group, ...]
    members: tuple[MemberRule, ...]
    group_members: tuple[GroupMember, ...]
    metrics: tuple[MetricDefinition, ...]
    group_metrics: tuple[GroupMetric, ...]
    metric_dependencies: tuple[MetricDependency, ...] = ()


@dataclass(frozen=True)
class ResolvedBranch:
    selected_group_id: str
    source_group_id: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedMember:
    selected_group_id: str
    source_group_id: str
    matched_member_id: str
    team_position: str | None
    champion_id: int | None
    build_scope: str | None
    specificity: int


@dataclass(frozen=True)
class ResolvedMetric:
    selected_group_id: str
    source_group_id: str
    metric_id: str
    metric_name: str
    metric_kind: MetricKind
    default_aggregation: Aggregation
    declared_by_group_id: str


@dataclass(frozen=True)
class DynamicMetricExecutionPlan:
    selected_group_id: str
    branches: tuple[ResolvedBranch, ...]
    members: tuple[ResolvedMember, ...]
    metrics: tuple[ResolvedMetric, ...]
    metric_dependencies: tuple[MetricDependency, ...]


def build_execution_plan(
    snapshot: MetadataSnapshot,
    *,
    selected_group_id: str,
) -> DynamicMetricExecutionPlan:
    groups_by_id = {
        group.group_id: group for group in snapshot.groups if group.is_active
    }
    if selected_group_id not in groups_by_id:
        raise KeyError(f"Unknown selected_group_id: {selected_group_id}")

    children_by_parent: dict[str, list[str]] = {}
    for group in groups_by_id.values():
        if group.parent_group_id is None:
            continue
        children_by_parent.setdefault(group.parent_group_id, []).append(group.group_id)

    members_by_id = {
        member.member_id: member for member in snapshot.members if member.is_active
    }
    group_members_by_group: dict[str, list[GroupMember]] = {}
    for group_member in snapshot.group_members:
        if group_member.is_active:
            group_members_by_group.setdefault(group_member.group_id, []).append(
                group_member
            )

    metrics_by_id = {
        metric.metric_id: metric for metric in snapshot.metrics if metric.is_active
    }
    group_metrics_by_group: dict[str, list[GroupMetric]] = {}
    for group_metric in snapshot.group_metrics:
        if group_metric.is_active:
            group_metrics_by_group.setdefault(group_metric.group_id, []).append(
                group_metric
            )

    branches = _resolve_branches(
        selected_group_id=selected_group_id,
        children_by_parent=children_by_parent,
    )
    resolved_members = _resolve_members(
        branches=branches,
        group_members_by_group=group_members_by_group,
        members_by_id=members_by_id,
    )
    resolved_metrics = _resolve_metrics(
        branches=branches,
        group_metrics_by_group=group_metrics_by_group,
        metrics_by_id=metrics_by_id,
    )

    metric_ids = {metric.metric_id for metric in resolved_metrics}
    metric_dependencies = tuple(
        dependency
        for dependency in snapshot.metric_dependencies
        if dependency.metric_id in metric_ids
    )

    return DynamicMetricExecutionPlan(
        selected_group_id=selected_group_id,
        branches=branches,
        members=resolved_members,
        metrics=resolved_metrics,
        metric_dependencies=metric_dependencies,
    )


def _resolve_branches(
    *,
    selected_group_id: str,
    children_by_parent: dict[str, list[str]],
) -> tuple[ResolvedBranch, ...]:
    branches: list[ResolvedBranch] = []

    def walk(group_id: str, path: tuple[str, ...]) -> None:
        if group_id in path:
            cycle = " -> ".join((*path, group_id))
            raise ValueError(f"Group cycle detected: {cycle}")

        branch_path = (*path, group_id)
        branches.append(
            ResolvedBranch(
                selected_group_id=selected_group_id,
                source_group_id=group_id,
                path=branch_path,
            )
        )

        for child_group_id in children_by_parent.get(group_id, ()):
            walk(child_group_id, branch_path)

    walk(selected_group_id, ())
    return tuple(branches)


def _resolve_members(
    *,
    branches: tuple[ResolvedBranch, ...],
    group_members_by_group: dict[str, list[GroupMember]],
    members_by_id: dict[str, MemberRule],
) -> tuple[ResolvedMember, ...]:
    resolved: list[ResolvedMember] = []

    for branch in branches:
        deduped: dict[str, ResolvedMember] = {}
        for group_id in branch.path:
            for group_member in group_members_by_group.get(group_id, ()):
                member = members_by_id.get(group_member.member_id)
                if member is None:
                    continue
                resolved_member = ResolvedMember(
                    selected_group_id=branch.selected_group_id,
                    source_group_id=branch.source_group_id,
                    matched_member_id=member.member_id,
                    team_position=member.team_position,
                    champion_id=member.champion_id,
                    build_scope=member.build_scope,
                    specificity=_member_specificity(member),
                )
                current = deduped.get(member.member_id)
                if current is None or resolved_member.specificity > current.specificity:
                    deduped[member.member_id] = resolved_member
        resolved.extend(
            sorted(
                deduped.values(),
                key=lambda item: (-item.specificity, item.matched_member_id),
            )
        )

    return tuple(resolved)


def _resolve_metrics(
    *,
    branches: tuple[ResolvedBranch, ...],
    group_metrics_by_group: dict[str, list[GroupMetric]],
    metrics_by_id: dict[str, MetricDefinition],
) -> tuple[ResolvedMetric, ...]:
    resolved: list[ResolvedMetric] = []

    for branch in branches:
        effective_by_metric_id: dict[str, ResolvedMetric] = {}
        for group_id in branch.path:
            for group_metric in group_metrics_by_group.get(group_id, ()):
                metric = metrics_by_id.get(group_metric.metric_id)
                if metric is None:
                    continue
                effective_by_metric_id[metric.metric_id] = ResolvedMetric(
                    selected_group_id=branch.selected_group_id,
                    source_group_id=branch.source_group_id,
                    metric_id=metric.metric_id,
                    metric_name=metric.metric_name,
                    metric_kind=metric.metric_kind,
                    default_aggregation=metric.default_aggregation,
                    declared_by_group_id=group_id,
                )
        resolved.extend(
            sorted(effective_by_metric_id.values(), key=lambda item: item.metric_id)
        )

    return tuple(resolved)


def _member_specificity(member: MemberRule) -> int:
    score = 0
    if member.team_position is not None:
        score += 1
    if member.champion_id is not None:
        score += 1
    if member.build_scope is not None:
        score += 1
    return score
