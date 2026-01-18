from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeVar

from pydantic import NonNegativeInt

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


class EventParser(Protocol[InT, OutT]):
    def parse(self, validated: InT) -> OutT: ...


POutT = TypeVar("POutT", covariant=True)


class ParticipantParser(Protocol[POutT]):
    def parse(
        self,
        participants: Sequence[Any],
        gameId: NonNegativeInt,
    ) -> POutT: ...


RawT = TypeVar("RawT", contravariant=True)
RunOutT = TypeVar("RunOutT", covariant=True)
