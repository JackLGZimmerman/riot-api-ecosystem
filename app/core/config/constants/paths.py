from __future__ import annotations

from pathlib import Path


# Root
def lake_dir() -> Path:
    return Path("lake")


def pipelines_dir() -> Path:
    return lake_dir() / "pipelines"


def pipeline_dir(name: str) -> Path:
    return pipelines_dir() / name


def dataset_dir(pipeline: str, dataset: str) -> Path:
    return pipeline_dir(pipeline) / dataset


def data_file(pipeline: str, dataset: str, filename: str) -> Path:
    return dataset_dir(pipeline, dataset) / filename


def shard_file(dir_: Path, idx: int, suffix: str) -> Path:
    return dir_ / f"part-{idx:06d}.{suffix}"


# ---- Players ----
def players_info_file() -> Path:
    return data_file("players", "info", "data.jsonl.zst")


def players_puuids_file() -> Path:
    return data_file("players", "puuids", "data.txt.zst")


# ---- Match IDs ----
def matchids_ids_dir() -> Path:
    return dataset_dir("matchids", "ids")


def matchids_puuids_file() -> Path:
    return data_file("matchids", "puuids", "data.txt.zst")


def matchids_puuids_checkpoint_file() -> Path:
    return data_file("matchids", "puuids", "checkpoint.json")


# ---- Match Data ----
def matchdata_matchids_file() -> Path:
    return dataset_dir("matchdata", "ids")


def matchdata_nontimeline_dir() -> Path:
    return dataset_dir("matchdata", "nontimeline")


def matchdata_timeline_dir() -> Path:
    return dataset_dir("matchdata", "timeline")


PLAYER_INFO = players_info_file()
PLAYER_PUUIDS = players_puuids_file()

MATCH_IDS_DATA_DIR = matchids_ids_dir()
PUUIDS_FOR_MATCH_IDS = matchids_puuids_file()
PUUIDS_FOR_MATCH_IDS_CHECKPOINT = matchids_puuids_checkpoint_file()

MATCH_DATA_DIR = pipeline_dir("matchdata")
MATCH_DATA_MATCH_IDS_DIR = matchdata_matchids_file()
MATCH_DATA_NO_TIMELINE_PATH = matchdata_nontimeline_dir()
MATCH_DATA_TIMELINE_PATH = matchdata_timeline_dir()
