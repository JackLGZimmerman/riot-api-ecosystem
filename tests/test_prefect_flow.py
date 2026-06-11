from __future__ import annotations

from app.worker.pipelines import prefect_flow


class DummyFactory:
    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs


def _patch_pipeline_factories(monkeypatch):
    created: list[str] = []

    class FakeOrchestrator:
        def __init__(self, *, pipeline: str, **kwargs) -> None:
            _ = kwargs
            self.pipeline = pipeline
            created.append(pipeline)

        async def run(self) -> None:
            return None

    for name in (
        "PlayersOrchestrator",
        "MatchIDOrchestrator",
        "MatchDataOrchestrator",
    ):
        monkeypatch.setattr(prefect_flow, name, FakeOrchestrator)

    for name in (
        "PlayerLoader",
        "PlayerCollector",
        "PlayerSaver",
        "MatchIDLoader",
        "MatchIDCollector",
        "MatchIDSaver",
        "MatchDataLoader",
        "MatchDataStreamCollector",
        "MatchDataSaver",
        "MatchDataNonTimelineParsingOrchestrator",
        "MatchDataTimelineParsingOrchestrator",
    ):
        monkeypatch.setattr(prefect_flow, name, DummyFactory)

    return created


def test_build_steps_defaults_to_full_pipeline(monkeypatch) -> None:
    created = _patch_pipeline_factories(monkeypatch)

    steps = prefect_flow._build_steps(object())

    assert [step.name for step in steps] == ["players", "match_ids", "match_data"]
    assert created == ["players", "match_ids", "match_data"]


def test_build_steps_matchdata_only_skips_upstream_stages(monkeypatch) -> None:
    created = _patch_pipeline_factories(monkeypatch)

    steps = prefect_flow._build_steps(object(), matchdata_only=True)

    assert [step.name for step in steps] == ["match_data"]
    assert created == ["match_data"]
