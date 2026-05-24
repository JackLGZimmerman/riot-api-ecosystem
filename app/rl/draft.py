"""Tournament-style pick/ban draft sequence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Side(IntEnum):
    BLUE = 0
    RED = 1


class ActionType(IntEnum):
    BAN = 0
    PICK = 1


@dataclass(frozen=True)
class DraftStep:
    index: int
    side: Side
    action_type: ActionType
    label: str


# Tournament order:
#   BB1, RB1, BB2, RB2, BB3, RB3,
#   B1, R1/R2, B2/B3, R3,
#   RB4, BB4, RB5, BB5,
#   R4, B4/B5, R5
DRAFT_SEQUENCE: tuple[DraftStep, ...] = (
    DraftStep(0, Side.BLUE, ActionType.BAN, "BB1"),
    DraftStep(1, Side.RED, ActionType.BAN, "RB1"),
    DraftStep(2, Side.BLUE, ActionType.BAN, "BB2"),
    DraftStep(3, Side.RED, ActionType.BAN, "RB2"),
    DraftStep(4, Side.BLUE, ActionType.BAN, "BB3"),
    DraftStep(5, Side.RED, ActionType.BAN, "RB3"),
    DraftStep(6, Side.BLUE, ActionType.PICK, "B1"),
    DraftStep(7, Side.RED, ActionType.PICK, "R1"),
    DraftStep(8, Side.RED, ActionType.PICK, "R2"),
    DraftStep(9, Side.BLUE, ActionType.PICK, "B2"),
    DraftStep(10, Side.BLUE, ActionType.PICK, "B3"),
    DraftStep(11, Side.RED, ActionType.PICK, "R3"),
    DraftStep(12, Side.RED, ActionType.BAN, "RB4"),
    DraftStep(13, Side.BLUE, ActionType.BAN, "BB4"),
    DraftStep(14, Side.RED, ActionType.BAN, "RB5"),
    DraftStep(15, Side.BLUE, ActionType.BAN, "BB5"),
    DraftStep(16, Side.RED, ActionType.PICK, "R4"),
    DraftStep(17, Side.BLUE, ActionType.PICK, "B4"),
    DraftStep(18, Side.BLUE, ActionType.PICK, "B5"),
    DraftStep(19, Side.RED, ActionType.PICK, "R5"),
)
