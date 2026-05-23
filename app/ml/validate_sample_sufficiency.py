"""Validate champion/position/build sample sufficiency for ML win-rate priors.

Run with:
    uv run python -m app.ml.validate_sample_sufficiency
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

from app.core.config.settings import PROJECT_ROOT
from app.core.logging.logger import setup_logging_config
from app.ml.config import ML_DATA_DIR, PLAYER_PIVOT_TABLE
from database.clickhouse.client import get_client

REPORT_PATH = ML_DATA_DIR / "sample_sufficiency_latest.json"
FAILURES_CSV_PATH = ML_DATA_DIR / "sample_sufficiency_failures.csv"
POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")

setup_logging_config()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SufficiencyConfig:
    power: float = 0.8
    alpha: float = 0.05
    detectable_effect: float = 0.05
    learning_buckets: int = 20
    learning_tail_fraction: float = 0.5
    max_learning_tail_delta: float = 0.03


@dataclass(frozen=True)
class CombinationCounts:
    championid: int
    championname: str
    teamposition: str
    build: str
    train_n: int
    train_wins: int
    val_n: int
    test_n: int
    pulled_n: int


@dataclass(frozen=True)
class LearningCurveMetrics:
    max_tail_delta: float | None
    tail_start_sample: int | None


@dataclass(frozen=True)
class CombinationAssessment:
    championid: int
    championname: str
    teamposition: str
    build: str
    train_n: int
    train_wins: int
    train_losses: int
    val_n: int
    test_n: int
    pulled_n: int
    win_rate: float | None
    power_required_n: int
    learning_curve: LearningCurveMetrics
    passed_checks: list[str]
    failed_checks: list[str]
    sufficient: bool


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _normal_quantile(probability: float) -> float:
    return NormalDist().inv_cdf(probability)


def _power_required_n(alpha: float, power: float, detectable_effect: float) -> int:
    if detectable_effect <= 0.0:
        raise ValueError("detectable_effect must be positive")

    p0 = 0.5
    p1 = min(1.0, p0 + detectable_effect)
    z_alpha = _normal_quantile(1.0 - alpha / 2.0)
    z_power = _normal_quantile(power)
    numerator = z_alpha * math.sqrt(p0 * (1.0 - p0)) + z_power * math.sqrt(
        p1 * (1.0 - p1)
    )
    return math.ceil((numerator / detectable_effect) ** 2)


def _learning_curve_metrics(
    bucket_counts: list[tuple[int, int]],
    total_wins: int,
    total_n: int,
    tail_fraction: float,
) -> LearningCurveMetrics:
    if total_n <= 0:
        return LearningCurveMetrics(
            max_tail_delta=None,
            tail_start_sample=None,
        )

    final_rate = total_wins / total_n
    tail_start_sample = math.ceil(total_n * tail_fraction)
    cumulative_wins = 0
    cumulative_n = 0
    max_tail_delta: float | None = None
    for wins, n in bucket_counts:
        cumulative_wins += wins
        cumulative_n += n
        if cumulative_n <= 0:
            continue
        rate = cumulative_wins / cumulative_n
        if cumulative_n >= tail_start_sample:
            delta = abs(rate - final_rate)
            max_tail_delta = (
                delta if max_tail_delta is None else max(max_tail_delta, delta)
            )

    return LearningCurveMetrics(
        max_tail_delta=max_tail_delta,
        tail_start_sample=tail_start_sample,
    )


def _pulled_samples_cte() -> str:
    return f"""
    pulled AS (
        SELECT
            p.matchid AS matchid,
            p.split AS split,
            tupleElement(tupleElement(token, 2), 1) AS championid,
            tupleElement(tupleElement(token, 2), 2) AS teamposition,
            tupleElement(tupleElement(token, 2), 3) AS build,
            toUInt8(
                if(tupleElement(token, 1) < 5, p.blue_win, 1 - p.blue_win)
            ) AS win
        FROM {PLAYER_PIVOT_TABLE} AS p
        ARRAY JOIN arrayZip(
            range(10),
            arrayConcat(p.blue_players, p.red_players)
        ) AS token
        WHERE p.split IN ('train', 'validation', 'test')
    )
    """


def _fetch_combination_counts() -> list[CombinationCounts]:
    query = f"""
    WITH
    {_pulled_samples_cte()}
    SELECT
        pulled.championid,
        dictGetOrDefault(
            'game_data.championid_name_map_dict',
            'name',
            toString(pulled.championid),
            ''
        ) AS championname,
        pulled.teamposition,
        pulled.build,
        countIf(pulled.split = 'train') AS train_n,
        sumIf(pulled.win, pulled.split = 'train') AS train_wins,
        countIf(pulled.split = 'validation') AS val_n,
        countIf(pulled.split = 'test') AS test_n,
        count() AS pulled_n
    FROM pulled
    GROUP BY
        pulled.championid,
        pulled.teamposition,
        pulled.build
    ORDER BY
        pulled.teamposition,
        pulled.championid,
        pulled.build
    """
    rows = get_client().query(query).result_rows
    return [
        CombinationCounts(
            championid=int(row[0]),
            championname=str(row[1]),
            teamposition=str(row[2]),
            build=str(row[3]),
            train_n=int(row[4]),
            train_wins=int(row[5]),
            val_n=int(row[6]),
            test_n=int(row[7]),
            pulled_n=int(row[8]),
        )
        for row in rows
    ]


def _fetch_partition_counts(
    *,
    partitions: int,
    partition_expr: str,
) -> dict[tuple[int, str, str], list[tuple[int, int]]]:
    query = f"""
    WITH
    {_pulled_samples_cte()}
    SELECT
        championid,
        teamposition,
        build,
        {partition_expr} AS partition_idx,
        count() AS n,
        sum(win) AS wins
    FROM pulled
    WHERE split = 'train'
    GROUP BY
        championid,
        teamposition,
        build,
        partition_idx
    ORDER BY
        championid,
        teamposition,
        build,
        partition_idx
    """
    result: dict[tuple[int, str, str], list[tuple[int, int]]] = {}
    for championid, teamposition, build, partition_idx, n, wins in (
        get_client().query(query).result_rows
    ):
        key = (int(championid), str(teamposition), str(build))
        if key not in result:
            result[key] = [(0, 0)] * partitions
        result[key][int(partition_idx)] = (int(wins), int(n))
    return result


def _status_checks(
    counts: CombinationCounts,
    power_required_n: int,
    learning_curve: LearningCurveMetrics,
    cfg: SufficiencyConfig,
) -> tuple[list[str], list[str]]:
    checks = {
        "power_required_n": counts.train_n >= power_required_n,
        "learning_curve_tail_delta": (
            learning_curve.max_tail_delta is not None
            and learning_curve.max_tail_delta <= cfg.max_learning_tail_delta
        ),
    }
    passed = [name for name, passed_check in checks.items() if passed_check]
    failed = [name for name, passed_check in checks.items() if not passed_check]
    return passed, failed


def _assess_combinations(
    counts: list[CombinationCounts],
    bucket_counts: dict[tuple[int, str, str], list[tuple[int, int]]],
    cfg: SufficiencyConfig,
) -> list[CombinationAssessment]:
    power_required_n = _power_required_n(
        alpha=cfg.alpha,
        power=cfg.power,
        detectable_effect=cfg.detectable_effect,
    )

    assessments: list[CombinationAssessment] = []
    for item in counts:
        key = (item.championid, item.teamposition, item.build)
        train_losses = item.train_n - item.train_wins
        learning_curve = _learning_curve_metrics(
            bucket_counts=bucket_counts.get(key, []),
            total_wins=item.train_wins,
            total_n=item.train_n,
            tail_fraction=cfg.learning_tail_fraction,
        )
        passed_checks, failed_checks = _status_checks(
            counts=item,
            power_required_n=power_required_n,
            learning_curve=learning_curve,
            cfg=cfg,
        )
        assessments.append(
            CombinationAssessment(
                championid=item.championid,
                championname=item.championname,
                teamposition=item.teamposition,
                build=item.build,
                train_n=item.train_n,
                train_wins=item.train_wins,
                train_losses=train_losses,
                val_n=item.val_n,
                test_n=item.test_n,
                pulled_n=item.pulled_n,
                win_rate=(item.train_wins / item.train_n if item.train_n > 0 else None),
                power_required_n=power_required_n,
                learning_curve=learning_curve,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                sufficient=not failed_checks,
            )
        )

    return assessments


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return _project_relative(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _summarize(
    assessments: list[CombinationAssessment],
    cfg: SufficiencyConfig,
) -> dict[str, Any]:
    total = len(assessments)
    sufficient = sum(assessment.sufficient for assessment in assessments)
    failure_counts: dict[str, int] = {}
    for assessment in assessments:
        for failed_check in assessment.failed_checks:
            failure_counts[failed_check] = failure_counts.get(failed_check, 0) + 1

    by_position: dict[str, dict[str, int]] = {}
    for position in POSITIONS:
        position_items = [
            assessment
            for assessment in assessments
            if assessment.teamposition == position
        ]
        by_position[position] = {
            "total": len(position_items),
            "sufficient": sum(item.sufficient for item in position_items),
            "insufficient": sum(not item.sufficient for item in position_items),
        }

    train_counts = sorted(assessment.train_n for assessment in assessments)
    return {
        "total_combinations": total,
        "sufficient_combinations": sufficient,
        "insufficient_combinations": total - sufficient,
        "all_combinations_sufficient": sufficient == total,
        "failure_counts": dict(sorted(failure_counts.items())),
        "by_position": by_position,
        "train_sample_count": {
            "min": train_counts[0] if train_counts else 0,
            "median": train_counts[len(train_counts) // 2] if train_counts else 0,
            "max": train_counts[-1] if train_counts else 0,
        },
        "power_required_n": _power_required_n(
            alpha=cfg.alpha,
            power=cfg.power,
            detectable_effect=cfg.detectable_effect,
        ),
    }


def _write_report(
    *,
    report_path: Path,
    failures_csv_path: Path,
    cfg: SufficiencyConfig,
    summary: dict[str, Any],
    assessments: list[CombinationAssessment],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "summary": summary,
        "combinations": assessments,
    }
    report_path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    failures = [assessment for assessment in assessments if not assessment.sufficient]
    with failures_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "championid",
                "championname",
                "teamposition",
                "build",
                "train_n",
                "train_wins",
                "train_losses",
                "win_rate",
                "power_required_n",
                "learning_max_tail_delta",
                "failed_checks",
            ]
        )
        for item in sorted(failures, key=lambda assessment: assessment.train_n):
            writer.writerow(
                [
                    item.championid,
                    item.championname,
                    item.teamposition,
                    item.build,
                    item.train_n,
                    item.train_wins,
                    item.train_losses,
                    item.win_rate,
                    item.power_required_n,
                    item.learning_curve.max_tail_delta,
                    ",".join(item.failed_checks),
                ]
            )


def validate(
    cfg: SufficiencyConfig | None = None,
    *,
    report_path: Path = REPORT_PATH,
    failures_csv_path: Path = FAILURES_CSV_PATH,
) -> dict[str, Any]:
    cfg = cfg or SufficiencyConfig()
    combination_counts = _fetch_combination_counts()
    bucket_counts = _fetch_partition_counts(
        partitions=cfg.learning_buckets,
        partition_expr=f"toUInt8(cityHash64(matchid) % {cfg.learning_buckets})",
    )
    assessments = _assess_combinations(
        counts=combination_counts,
        bucket_counts=bucket_counts,
        cfg=cfg,
    )
    summary = _summarize(
        assessments=assessments,
        cfg=cfg,
    )
    _write_report(
        report_path=report_path,
        failures_csv_path=failures_csv_path,
        cfg=cfg,
        summary=summary,
        assessments=assessments,
    )

    logger.info("Combinations: %d", summary["total_combinations"])
    logger.info(
        "Sufficient: %d / %d",
        summary["sufficient_combinations"],
        summary["total_combinations"],
    )
    logger.info("Failure counts: %s", summary["failure_counts"])
    logger.info("Report: %s", _project_relative(report_path))
    logger.info("Failures CSV: %s", _project_relative(failures_csv_path))
    return {
        "summary": summary,
        "report_path": report_path,
        "failures_csv_path": failures_csv_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate sample sufficiency for champion/position/build win-rate priors."
        )
    )
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--failures-csv-path", type=Path, default=FAILURES_CSV_PATH)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--detectable-effect", type=float, default=0.05)
    parser.add_argument("--learning-buckets", type=int, default=20)
    parser.add_argument("--learning-tail-fraction", type=float, default=0.5)
    parser.add_argument("--max-learning-tail-delta", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = SufficiencyConfig(
        power=args.power,
        alpha=args.alpha,
        detectable_effect=args.detectable_effect,
        learning_buckets=args.learning_buckets,
        learning_tail_fraction=args.learning_tail_fraction,
        max_learning_tail_delta=args.max_learning_tail_delta,
    )
    validate(
        cfg,
        report_path=args.report_path,
        failures_csv_path=args.failures_csv_path,
    )


if __name__ == "__main__":
    main()
