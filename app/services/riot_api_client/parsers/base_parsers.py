from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar

from pydantic import NonNegativeInt

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


class ParticipantLike(Protocol):
    puuid: str
    participantId: int
    teamId: int
    championId: int
    championName: str

    kills: int
    deaths: int
    assists: int

    challenges: Any
    perks: Any


class InfoParser(Protocol[InT, OutT]):
    def parse(self, validated: InT, /) -> OutT: ...


class ParticipantParser(Protocol[OutT]):
    def parse(
        self,
        participants: Sequence[ParticipantLike],
        gameId: int,
    ) -> OutT: ...


class EventParser(Protocol[InT, OutT]):
    def parse(self, validated: InT, gameId: int, /) -> OutT: ...
