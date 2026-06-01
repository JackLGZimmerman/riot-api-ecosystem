"""HTML report for specialist embedding groups."""

from __future__ import annotations

import argparse
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
    SPECIALIST_CACHE_DIR,
    SPECIALISTS,
    SpecialistSpec,
)
from app.core.config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

CHAMPION_NAMES_PATH = (
    PROJECT_ROOT / "database" / "clickhouse" / "support" / "championid_name_map.jsonl"
)


@dataclass(frozen=True)
class SpecialistGroup:
    rank: int
    label_id: int
    members: list[int]
    builds: Counter[str]
    roles: Counter[str]
    champions: Counter[int]


@dataclass(frozen=True)
class SpecialistReport:
    name: str
    display_name: str
    feature_set: tuple[str, ...]
    groups: list[SpecialistGroup]
    unlabelled: list[int]
    keys: list[tuple]
    columns: dict[str, int]
    updated_at: datetime

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def unlabelled_count(self) -> int:
        return len(self.unlabelled)

    @property
    def coverage(self) -> float:
        total = len(self.keys)
        return (total - self.unlabelled_count) / total if total else float("nan")

    @property
    def largest_group(self) -> int:
        return max((len(group.members) for group in self.groups), default=0)


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt_label(label: str) -> str:
    return label.replace("_", " ")


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100.0 * value:.1f}%"


def _hue(label: str) -> int:
    return sum((i + 1) * ord(char) for i, char in enumerate(label)) % 360


def _style(label: str) -> str:
    return f"--h:{_hue(label)}"


def _anchor(name: str) -> str:
    return name.replace("_", "-").replace(".", "-")


def _normalise(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def _display_name(name: str) -> str:
    return _fmt_label(name).title()


def _load_champion_names(path: Path = CHAMPION_NAMES_PATH) -> dict[int, str]:
    if not path.exists():
        return {}
    names: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        names[int(row["_key"])] = str(row["name"])
    return names


def _required_columns(key_columns: tuple[str, ...], source: str) -> dict[str, int]:
    columns = {name: i for i, name in enumerate(key_columns)}
    missing = {"championid", "teamposition", "build"} - columns.keys()
    if missing:
        raise ValueError(f"{source} requires keys, missing {sorted(missing)}")
    return columns


def _counter(keys: list[tuple], members: list[int], columns: dict[str, int], name: str) -> Counter:
    return Counter(str(keys[index][columns[name]]) for index in members)


def _champion_counter(
    keys: list[tuple],
    members: list[int],
    columns: dict[str, int],
) -> Counter[int]:
    return Counter(int(keys[index][columns["championid"]]) for index in members)


def _chip(label: str, count: int | None = None) -> str:
    suffix = "" if count is None else f"<b>{count}</b>"
    return (
        f'<span class="chip" style="{_style(label)}">'
        f"{_html(_fmt_label(label))}{suffix}</span>"
    )


def _chips(values: Counter[str], limit: int | None = None) -> str:
    return "".join(_chip(label, count) for label, count in values.most_common(limit))


def _champion_label(championid: int, champion_names: dict[int, str]) -> str:
    return f"{champion_names.get(championid, championid)} #{championid}"


def _top_champions(
    champions: Counter[int],
    champion_names: dict[int, str],
    limit: int = 5,
) -> str:
    return ", ".join(
        f"{_html(_champion_label(championid, champion_names))}({count})"
        for championid, count in champions.most_common(limit)
    )


def _identity_chip(
    key: tuple,
    columns: dict[str, int],
    champion_names: dict[int, str],
) -> str:
    championid = int(key[columns["championid"]])
    build = str(key[columns["build"]])
    return (
        f'<span class="member" style="{_style(build)}">'
        f'<b>{_html(champion_names.get(championid, championid))}</b>'
        f'<span>#{championid}</span>'
        f'<span>{_html(key[columns["teamposition"]])}</span>'
        f"<span>{_html(_fmt_label(build))}</span></span>"
    )


def _members_html(
    keys: list[tuple],
    members: list[int],
    columns: dict[str, int],
    champion_names: dict[int, str],
) -> str:
    return "".join(_identity_chip(keys[index], columns, champion_names) for index in members)


def _groups_from_labels(
    labels: np.ndarray,
    keys: list[tuple],
    columns: dict[str, int],
) -> list[SpecialistGroup]:
    group_rows: list[tuple[int, list[int]]] = []
    for label in [v for v in np.unique(labels) if v >= 0]:
        members = np.flatnonzero(labels == label).astype(int).tolist()
        group_rows.append((int(label), members))
    group_rows.sort(key=lambda row: (-len(row[1]), row[0]))
    return [
        SpecialistGroup(
            rank=rank,
            label_id=label,
            members=members,
            builds=_counter(keys, members, columns, "build"),
            roles=_counter(keys, members, columns, "teamposition"),
            champions=_champion_counter(keys, members, columns),
        )
        for rank, (label, members) in enumerate(group_rows, start=1)
    ]


def _load_specialist_report(path: Path, spec: SpecialistSpec | None) -> SpecialistReport:
    with np.load(path, allow_pickle=True) as data:
        keys = [tuple(key) for key in data["keys"].tolist()]
        columns = _required_columns(
            tuple(str(column) for column in data["key_columns"].tolist()),
            f"{path.name}",
        )
        labels = data["labels"].astype(np.int32)
        if labels.ndim != 1:
            raise ValueError(
                f"{path.name} labels must be 1-D for non-temporal classification, "
                f"got {labels.shape}"
            )
    return SpecialistReport(
        name=path.stem,
        display_name=_display_name(path.stem),
        feature_set=spec.feature_set if spec else (),
        groups=_groups_from_labels(labels, keys, columns),
        unlabelled=np.flatnonzero(labels < 0).astype(int).tolist(),
        keys=keys,
        columns=columns,
        updated_at=datetime.fromtimestamp(path.stat().st_mtime),
    )


def _specialist_paths(specialist_dir: Path, focus: str | None) -> list[Path]:
    paths = sorted(specialist_dir.glob("*.npz"))
    if focus is None:
        by_name = {path.stem: path for path in paths}
        return [by_name[spec.name] for spec in SPECIALISTS if spec.name in by_name]

    wanted = _normalise(focus)
    matches = [
        path
        for path in paths
        if wanted in {_normalise(path.stem), _normalise(_display_name(path.stem))}
    ]
    if matches:
        return matches
    options = ", ".join(path.stem for path in paths)
    raise ValueError(f"No specialist matched {focus!r}. Available: {options}")


def _load_specialist_reports(
    specialist_dir: Path,
    focus: str | None = None,
) -> list[SpecialistReport]:
    if not specialist_dir.exists():
        return []
    specs = {spec.name: spec for spec in SPECIALISTS}
    return [
        _load_specialist_report(path, specs.get(path.stem))
        for path in _specialist_paths(specialist_dir, focus)
    ]


STYLE = """
<style>
:root { color-scheme: light; --bg:#f7f8fa; --panel:#fff; --text:#202831; --muted:#667085; --line:#d8dee8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 22px 54px; }
h1 { margin:0 0 6px; font-size:28px; }
h2 { margin:28px 0 10px; font-size:20px; }
p { margin:0 0 14px; color:var(--muted); }
a { color:#175cd3; text-decoration:none; }
nav, .kpis, .meta, .members, .feature-list { display:flex; flex-wrap:wrap; gap:8px; }
nav { margin:18px 0 22px; }
nav a, .kpi, details.group, table { background:var(--panel); border:1px solid var(--line); border-radius:8px; }
nav a { padding:5px 9px; border-radius:999px; color:var(--text); }
.kpis { margin:14px 0; }
.kpi { min-width:132px; padding:10px; }
.kpi b { display:block; font-size:20px; }
.kpi span, .small { color:var(--muted); font-size:12px; }
table { width:100%; border-collapse:collapse; overflow:hidden; }
th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:#eef1f5; color:#475467; font-size:12px; text-transform:uppercase; }
tr:last-child td { border-bottom:0; }
.section { margin-top:28px; padding-top:8px; border-top:2px solid var(--line); }
.group-list { display:grid; gap:9px; margin-top:10px; }
details.group { overflow:hidden; }
summary { cursor:pointer; display:flex; justify-content:space-between; gap:12px; padding:10px 12px; background:#fbfcfe; }
.group-title { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
.group-body { padding:10px 12px 12px; border-top:1px solid var(--line); }
.meta { margin-bottom:10px; color:var(--muted); }
.feature-list { margin:8px 0 10px; }
.chip, .badge, .member { --h:210; border:1px solid hsl(var(--h) 55% 75%); background:hsl(var(--h) 78% 96%); color:hsl(var(--h) 48% 30%); }
.chip, .member { display:inline-flex; align-items:center; gap:5px; border-radius:999px; padding:3px 7px; font-size:12px; }
.chip b { color:#344054; }
.badge { display:inline-flex; min-width:46px; justify-content:center; border-radius:6px; padding:3px 6px; font-weight:800; font-size:12px; }
.member { border-radius:6px; }
.member b { color:#202831; }
.member span { color:#667085; }
@media (max-width:820px) { main { padding:18px 12px 42px; } summary { flex-direction:column; } table { font-size:12px; } th, td { padding:7px 6px; } }
</style>
"""


def _page(title: str, intro: str, nav: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_html(title)}</title>{STYLE}</head><body><main>"
        f"<h1>{_html(title)}</h1>{intro}{nav}{body}</main></body></html>"
    )


def _kpis(values: list[tuple[str, str]]) -> str:
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><b>{_html(value)}</b><span>{_html(label)}</span></div>'
        for value, label in values
    ) + "</div>"


def _group_html(
    *,
    rank: int,
    badge: str,
    badge_style: str,
    count: int,
    champions: Counter[int],
    builds: Counter[str],
    roles: Counter[str],
    members: str,
    champion_names: dict[int, str],
    stats: str = "",
    open_group: bool = False,
) -> str:
    open_attr = " open" if open_group else ""
    right = stats or _top_champions(champions, champion_names)
    return (
        f'<details class="group"{open_attr}><summary><span class="group-title">'
        f'<span class="small">#{rank}</span>'
        f'<span class="badge" style="{badge_style}">{_html(badge)}</span>'
        f"<b>{count} identities</b>"
        f'<span class="small">{len(champions)} champions</span></span>'
        f'<span class="small">{right}</span></summary>'
        '<div class="group-body">'
        f'<div class="meta"><span>Builds: {_chips(builds, 8)}</span>'
        f"<span>Roles: {_chips(roles)}</span></div>"
        f'<div class="members">{members}</div></div></details>'
    )


def _specialist_table(reports: list[SpecialistReport]) -> str:
    rows = "".join(
        "<tr>"
        f'<td><a href="#specialist-{_anchor(report.name)}">{_html(report.display_name)}</a>'
        f'<div class="small">{_html(report.name)}</div></td>'
        f"<td>{report.n_groups}</td>"
        f"<td>{report.unlabelled_count}</td>"
        f"<td>{report.largest_group}</td><td>{_pct(report.coverage)}</td></tr>"
        for report in reports
    )
    return (
        "<table><thead><tr><th>Specialist</th><th>Groups</th>"
        "<th>Unlabelled</th><th>Largest</th><th>Coverage</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _group_list(
    report: SpecialistReport,
    champion_names: dict[int, str],
    *,
    open_groups: bool,
) -> str:
    groups = [
        _group_html(
            rank=group.rank,
            badge=f"G{group.label_id}",
            badge_style=_style(f"{report.name}{group.label_id}"),
            count=len(group.members),
            champions=group.champions,
            builds=group.builds,
            roles=group.roles,
            members=_members_html(report.keys, group.members, report.columns, champion_names),
            champion_names=champion_names,
            open_group=open_groups,
        )
        for group in report.groups
    ]
    if report.unlabelled:
        groups.append(
            _group_html(
                rank=len(report.groups) + 1,
                badge="-1",
                badge_style="",
                count=len(report.unlabelled),
                champions=_champion_counter(report.keys, report.unlabelled, report.columns),
                builds=_counter(report.keys, report.unlabelled, report.columns, "build"),
                roles=_counter(report.keys, report.unlabelled, report.columns, "teamposition"),
                members=_members_html(
                    report.keys, report.unlabelled, report.columns, champion_names
                ),
                champion_names=champion_names,
                stats=f"no coherent {report.display_name} read",
                open_group=open_groups,
            )
        )
    return f'<div class="group-list">{"".join(groups)}</div>'


def _specialist_section(
    report: SpecialistReport,
    champion_names: dict[int, str],
    *,
    open_groups: bool,
) -> str:
    return (
        f'<section class="section" id="specialist-{_anchor(report.name)}">'
        f"<h2>{_html(report.display_name)}</h2>"
        f'<p class="small">{_html(report.name)}.npz updated '
        f"{report.updated_at.strftime('%Y-%m-%d %H:%M')}</p>"
        + _kpis(
            [
                (str(report.n_groups), "groups"),
                (str(report.unlabelled_count), "unlabelled"),
                (_pct(report.coverage), "coverage"),
                (str(report.largest_group), "largest"),
            ]
        )
        + f'<div class="feature-list">{"".join(_chip(feature) for feature in report.feature_set)}</div>'
        + _group_list(report, champion_names, open_groups=open_groups)
        + "</section>"
    )


def _specialist_body(
    reports: list[SpecialistReport],
    champion_names: dict[int, str],
    *,
    focus: str | None,
) -> str:
    table = "" if focus else f"<h2>Summary</h2>{_specialist_table(reports)}"
    return table + "".join(
        _specialist_section(report, champion_names, open_groups=focus is not None)
        for report in reports
    )


def write_specialist_report(
    cfg: EmbeddingConfig,
    specialist_dir: Path = SPECIALIST_CACHE_DIR,
    champion_names_path: Path = CHAMPION_NAMES_PATH,
    *,
    focus: str | None = None,
    output_path: Path | None = None,
) -> Path:
    reports = _load_specialist_reports(specialist_dir, focus)
    if not reports:
        raise FileNotFoundError(f"No specialist embedding caches found in {specialist_dir}")

    title = (
        f"Classification Specialist: {reports[0].display_name}"
        if focus
        else "Classification Specialist Report"
    )
    target_path = output_path or (
        cfg.specialist_report_path.with_name(f"specialist_{reports[0].name}_report.html")
        if focus
        else cfg.specialist_report_path
    )
    if focus:
        nav = (
            f'<nav><a href="{_html(cfg.specialist_report_path.name)}">All specialists</a></nav>'
        )
    else:
        nav = "<nav>" + "".join(
            f'<a href="#specialist-{_anchor(report.name)}">{_html(report.display_name)}</a>'
            for report in reports
        ) + "</nav>"

    champion_names = _load_champion_names(champion_names_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        _page(
            title,
            (
                "<p>Each section asks one narrow behavioural question over the "
                "same <b>(championid, teamposition, build)</b> identities. "
                f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.</p>"
            ),
            nav,
            _specialist_body(reports, champion_names, focus=focus),
        ),
        encoding="utf-8",
    )
    logger.info("Wrote specialist report: %s", target_path)
    return target_path


def main() -> None:
    from app.core.logging.logger import setup_logging_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", "--focus", dest="focus")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    setup_logging_config()
    write_specialist_report(EmbeddingConfig(), focus=args.focus, output_path=args.output)


if __name__ == "__main__":
    main()
