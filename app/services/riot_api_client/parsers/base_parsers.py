from __future__ import annotations

from typing import Protocol, TypeVar
from collections.abc import Sequence

from app.services.riot_api_client.parsers.models.non_timeline import (
    Participant,
)

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


class InfoParser(Protocol[InT, OutT]):
    def parse(self, validated: InT, matchId: str | int, /) -> OutT: ...


class ParticipantParser(Protocol[OutT]):
    def parse(
        self,
        participants: Sequence[Participant],
        matchId: str | int,
    ) -> OutT: ...


class EventParser(Protocol[InT, OutT]):
    def parse(self, validated: InT, matchId: str | int, /) -> OutT: ...
