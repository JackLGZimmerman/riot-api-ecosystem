from __future__ import annotations

from typing import Protocol, Sequence, TypeVar

from app.services.riot_api_client.parsers.models.non_timeline import (
    Participant,
)

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


class InfoParser(Protocol[InT, OutT]):
    def parse(self, validated: InT, /) -> OutT: ...


class ParticipantParser(Protocol[OutT]):
    def parse(
        self,
        participants: Sequence[Participant],
        gameId: int,
    ) -> OutT: ...


class EventParser(Protocol[InT, OutT]):
    def parse(self, validated: InT, gameId: int, /) -> OutT: ...
