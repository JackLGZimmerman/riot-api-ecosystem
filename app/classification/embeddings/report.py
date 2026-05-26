"""HTML report for baseline embedding groups."""

from __future__ import annotations

import html
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from app.classification.embeddings.config import (
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.embed import LevelEmbeddings
from app.classification.embeddings.similarity import (
    cosine_similarity_matrix,
    group_by_threshold,
)
from app.core.config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

CHAMPION_NAMES_PATH = (
    PROJECT_ROOT / "database" / "clickhouse" / "support" / "championid_name_map.jsonl"
)


@dataclass(frozen=True)
class GroupSummary:
    rank: int
    members: list[int]
    label: str
    builds: Counter[str]
    roles: Counter[str]
    champions: Counter[int]
    pairwise: np.ndarray


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    groups: list[GroupSummary]
    singletons: list[int]
    coverage: float
    dominant_build_share: float
    within_pairwise: np.ndarray

    @property
    def group_count(self) -> int:
        return len(self.groups) + len(self.singletons)

    @property
    def largest_group(self) -> int:
        return max(
            (len(g.members) for g in self.groups), default=1 if self.singletons else 0
        )


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)


def _hue(label: str) -> int:
    return sum((i + 1) * ord(char) for i, char in enumerate(label)) % 360


def _style(label: str) -> str:
    return f"--h:{_hue(label)}"


def _fmt_label(label: str) -> str:
    return label.replace("_", " ")


def _fmt_float(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.1f}%"


def _threshold_id(threshold: float) -> str:
    return f"threshold-{threshold:g}".replace(".", "-")


def _thresholds(cfg: EmbeddingConfig) -> tuple[float, ...]:
    return (float(cfg.similarity_threshold),)


def _load_champion_names(path: Path = CHAMPION_NAMES_PATH) -> dict[int, str]:
    if not path.exists():
        return {}
    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        names[int(row["_key"])] = str(row["name"])
    return names


def _key_columns(embeddings: LevelEmbeddings) -> dict[str, int]:
    required = {"championid", "teamposition", "build"}
    columns = {name: i for i, name in enumerate(embeddings.key_columns)}
    missing = required - columns.keys()
    if missing:
        raise ValueError(
            f"grouping report requires baseline keys, missing {sorted(missing)}"
        )
    return columns


def _pairwise_values(sim: np.ndarray, members: list[int]) -> np.ndarray:
    if len(members) < 2:
        return np.array([], dtype=np.float32)
    idx = np.asarray(members, dtype=np.int64)
    left, right = np.triu_indices(idx.size, k=1)
    return sim[idx[left], idx[right]]


def _group_label(builds: Counter[str]) -> str:
    ranked = builds.most_common()
    if not ranked:
        return "unknown"
    top_build, top_count = ranked[0]
    total = builds.total()
    if len(ranked) == 1 or top_count / total >= 0.6:
        return _fmt_label(top_build)
    kept = [build for build, count in ranked[:3] if count / total >= 0.15]
    return " + ".join(_fmt_label(build) for build in (kept or [top_build]))


def _chip(label: str, count: int | None = None) -> str:
    count_html = "" if count is None else f"<b>{count}</b>"
    return (
        f'<span class="chip" style="{_style(label)}">'
        f"{_html(_fmt_label(label))}{count_html}</span>"
    )


def _chips(counter: Counter[str], limit: int | None = None) -> str:
    ranked = counter.most_common(limit)
    return "".join(_chip(label, count) for label, count in ranked)


def _top_champions(
    champions: Counter[int], names: dict[int, str], limit: int = 8
) -> str:
    return ", ".join(
        f"{_html(names.get(championid, championid))}({count})"
        for championid, count in champions.most_common(limit)
    )


def _member_html(
    key: tuple,
    columns: dict[str, int],
    champion_names: dict[int, str],
) -> str:
    championid = int(key[columns["championid"]])
    build = str(key[columns["build"]])
    return (
        f'<span class="member" style="{_style(build)}">'
        f'<span class="name">{_html(champion_names.get(championid, championid))}</span>'
        f'<span class="role">{_html(key[columns["teamposition"]])}</span>'
        f"<span>{_html(_fmt_label(build))}</span></span>"
    )


def _summarise_group(
    rank: int,
    members: list[int],
    embeddings: LevelEmbeddings,
    columns: dict[str, int],
    sim: np.ndarray,
) -> GroupSummary:
    keys = [embeddings.keys[i] for i in members]
    builds = Counter(str(key[columns["build"]]) for key in keys)
    return GroupSummary(
        rank=rank,
        members=members,
        label=_group_label(builds),
        builds=builds,
        roles=Counter(str(key[columns["teamposition"]]) for key in keys),
        champions=Counter(int(key[columns["championid"]]) for key in keys),
        pairwise=_pairwise_values(sim, members),
    )


def _threshold_report(
    embeddings: LevelEmbeddings,
    threshold: float,
    columns: dict[str, int],
    sim: np.ndarray,
) -> ThresholdReport:
    raw_groups = group_by_threshold(embeddings.embeddings, threshold)
    non_singletons = [group for group in raw_groups if len(group) > 1]
    groups = [
        _summarise_group(i, group, embeddings, columns, sim)
        for i, group in enumerate(non_singletons, start=1)
    ]
    covered = sum(len(group.members) for group in groups)
    dominant = sum(max(group.builds.values()) for group in groups)
    pairwise = [group.pairwise for group in groups if group.pairwise.size]
    return ThresholdReport(
        threshold=threshold,
        groups=groups,
        singletons=[group[0] for group in raw_groups if len(group) == 1],
        coverage=covered / len(embeddings.keys) if embeddings.keys else float("nan"),
        dominant_build_share=dominant / covered if covered else float("nan"),
        within_pairwise=np.concatenate(pairwise) if pairwise else np.array([]),
    )


def _summary_table(reports: list[ThresholdReport]) -> str:
    rows = []
    for report in reports:
        values = report.within_pairwise
        p05, p25, p50 = (
            np.percentile(values, [5, 25, 50]) if values.size else [float("nan")] * 3
        )
        worst = min(
            (float(g.pairwise.min()) for g in report.groups), default=float("nan")
        )
        rows.append(
            "<tr>"
            f'<td><a href="#{_threshold_id(report.threshold)}">{report.threshold:g}</a></td>'
            f"<td>{report.group_count}</td><td>{len(report.singletons)}</td>"
            f"<td>{report.largest_group}</td><td>{_pct(report.coverage)}</td>"
            f"<td>{_pct(report.dominant_build_share)}</td>"
            f"<td>{_fmt_float(float(p05))} / {_fmt_float(float(p25))} / {_fmt_float(float(p50))}</td>"
            f"<td>{_fmt_float(worst)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Threshold</th><th>Groups</th><th>Singletons</th>"
        "<th>Largest</th><th>Non-singleton coverage</th><th>Dominant build</th>"
        "<th>Within sim p05 / p25 / median</th><th>Worst min sim</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _group_details(
    group: GroupSummary,
    threshold: float,
    embeddings: LevelEmbeddings,
    columns: dict[str, int],
    champion_names: dict[int, str],
) -> str:
    below = float((group.pairwise < threshold).mean()) if group.pairwise.size else 0.0
    members = "".join(
        _member_html(embeddings.keys[i], columns, champion_names) for i in group.members
    )
    return (
        '<details class="group">'
        '<summary><span class="group-title">'
        f'<span class="small">#{group.rank}</span>'
        f'<span class="badge" style="{_style(group.label)}">{_html(group.label)}</span>'
        f"<b>{len(group.members)} identities</b>"
        f'<span class="small">{len(group.champions)} champions</span></span>'
        f'<span class="small">median {_fmt_float(float(np.median(group.pairwise)))}'
        f" | min {_fmt_float(float(group.pairwise.min()))}"
        f" | below threshold {_pct(below)}</span></summary>"
        '<div class="group-body"><div class="meta">'
        f"<span>Builds: {_chips(group.builds)}</span>"
        f"<span>Roles: {_chips(group.roles)}</span></div>"
        f'<div class="members">{members}</div></div></details>'
    )


def _singletons_block(
    singletons: list[int],
    embeddings: LevelEmbeddings,
    columns: dict[str, int],
    champion_names: dict[int, str],
) -> str:
    if not singletons:
        return ""
    members = "".join(
        _member_html(embeddings.keys[i], columns, champion_names) for i in singletons
    )
    return (
        '<details class="group"><summary><span class="group-title">'
        f'<span class="badge">SINGLE</span><b>{len(singletons)} singleton identities</b>'
        f'</span></summary><div class="group-body"><div class="members">{members}'
        "</div></div></details>"
    )


def _primary_section(
    primary: ThresholdReport,
    embeddings: LevelEmbeddings,
    columns: dict[str, int],
    champion_names: dict[int, str],
) -> str:
    values = primary.within_pairwise
    median_sim = _fmt_float(float(np.median(values))) if values.size else "n/a"
    groups = "".join(
        _group_details(g, primary.threshold, embeddings, columns, champion_names)
        for g in primary.groups
    )
    return (
        f'<section class="section" id="{_threshold_id(primary.threshold)}">'
        f"<h2>Groups at threshold {primary.threshold:g}</h2>"
        '<div class="kpi-grid">'
        f'<div class="kpi"><b>{primary.group_count}</b><span>groups</span></div>'
        f'<div class="kpi"><b>{len(primary.singletons)}</b><span>singletons</span></div>'
        f'<div class="kpi"><b>{_pct(primary.coverage)}</b><span>coverage</span></div>'
        f'<div class="kpi"><b>{median_sim}</b><span>median within-pair</span></div>'
        "</div>"
        f'<div class="group-list">{groups}'
        f"{_singletons_block(primary.singletons, embeddings, columns, champion_names)}"
        "</div></section>"
    )


STYLE = """
<style>
:root { color-scheme: light; --bg:#f6f7f9; --panel:#fff; --text:#1e252d; --muted:#667085; --line:#d9dee7; --soft:#eef1f5; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:1240px; margin:0 auto; padding:28px 24px 56px; }
h1 { margin:0 0 4px; font-size:28px; }
h2 { margin:32px 0 12px; font-size:20px; }
h3 { margin:22px 0 10px; font-size:16px; }
p { margin:0 0 16px; color:var(--muted); }
a { color:#175cd3; text-decoration:none; }
table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
th, td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:#eef1f5; font-size:12px; text-transform:uppercase; color:#475467; }
tr:last-child td { border-bottom:0; }
nav { display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 22px; }
nav a { display:inline-flex; align-items:center; gap:6px; padding:5px 9px; border:1px solid var(--line); border-radius:999px; background:var(--panel); color:var(--text); }
.kpi-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; margin:14px 0; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
.kpi b { display:block; font-size:20px; }
.kpi span, .small { color:var(--muted); font-size:12px; }
.section { margin-top:28px; padding-top:6px; border-top:2px solid var(--line); }
.group-list { display:grid; grid-template-columns:1fr; gap:10px; margin-top:10px; }
details.group { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
details.group > summary { cursor:pointer; display:flex; gap:10px; align-items:center; justify-content:space-between; padding:10px 12px; background:#fbfcfe; }
.group-title { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
.group-body { padding:10px 12px 12px; border-top:1px solid var(--line); }
.meta, .members { display:flex; flex-wrap:wrap; gap:6px; }
.meta { margin-bottom:10px; color:var(--muted); gap:8px; }
.chip, .badge, .member { --h:210; border:1px solid hsl(var(--h) 55% 75%); background:hsl(var(--h) 78% 96%); color:hsl(var(--h) 50% 30%); }
.chip { display:inline-flex; align-items:center; gap:5px; padding:3px 7px; border-radius:999px; font-size:12px; margin:2px 3px 2px 0; }
.chip b { font-weight:700; color:#344054; }
.badge { display:inline-flex; align-items:center; min-width:48px; justify-content:center; border-radius:6px; padding:3px 6px; font-weight:800; font-size:12px; }
.member { display:inline-flex; gap:5px; align-items:center; border-radius:6px; padding:4px 6px; font-size:12px; }
.member .name { font-weight:700; color:#1e252d; }
.member .role { color:var(--muted); }
@media (max-width:820px) { main { padding:18px 12px 42px; } .kpi-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } table { font-size:12px; } th, td { padding:7px 6px; } details.group > summary { align-items:flex-start; flex-direction:column; } }
</style>
"""


def _build_html(
    embeddings: LevelEmbeddings,
    reports: list[ThresholdReport],
    columns: dict[str, int],
    champion_names: dict[int, str],
    primary_threshold: float,
) -> str:
    primary = next(
        (r for r in reports if abs(r.threshold - primary_threshold) < 1e-9),
        reports[0],
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Classification Grouping Report</title>"
        f"{STYLE}</head><body><main>"
        "<h1>Classification Grouping Report</h1>"
        f"<p>Identity unit: <b>(championid, teamposition, build)</b>. "
        "Clustering: average-link agglomerative over cosine distance. "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.</p>"
        "<h2>Threshold summary</h2>"
        f"{_summary_table(reports)}"
        f"{_primary_section(primary, embeddings, columns, champion_names)}"
        "</main></body></html>"
    )


def write_grouping_report(
    embeddings: LevelEmbeddings,
    cfg: EmbeddingConfig,
    champion_names_path: Path = CHAMPION_NAMES_PATH,
) -> Path:
    if embeddings.level is not IdentityType.BASELINE:
        raise ValueError("grouping report currently expects baseline embeddings")
    columns = _key_columns(embeddings)
    sim = cosine_similarity_matrix(embeddings.embeddings)
    reports = [
        _threshold_report(embeddings, threshold, columns, sim)
        for threshold in _thresholds(cfg)
    ]
    champion_names = _load_champion_names(champion_names_path)
    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(
        _build_html(
            embeddings, reports, columns, champion_names, cfg.similarity_threshold
        ),
        encoding="utf-8",
    )
    logger.info("Wrote grouping report: %s", cfg.report_path)
    return cfg.report_path


def main() -> None:
    from app.classification.embeddings.embed import load
    from app.core.logging.logger import setup_logging_config

    setup_logging_config()
    cfg = EmbeddingConfig()
    baseline = load(cfg.cache_dir).get(IdentityType.BASELINE)
    if baseline is None:
        raise SystemExit(f"No baseline embedding cache found in {cfg.cache_dir}")
    write_grouping_report(baseline, cfg)


if __name__ == "__main__":
    main()
