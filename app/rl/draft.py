"""Tournament-style pick/ban draft sequence and the mutable draft state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


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


@dataclass
class DraftState:
    """Mutable draft state shared by the env and MCTS; cheap to clone.

    Picks/bans are positional champion indices; the int8 ``available``
    vector is the legal mask (1 = selectable). Real champion ids are
    resolved only at the predictor boundary.
    """

    n_champions: int
    blue_picks: list[int] = field(default_factory=list)
    red_picks: list[int] = field(default_factory=list)
    blue_bans: list[int] = field(default_factory=list)
    red_bans: list[int] = field(default_factory=list)
    available: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    step_idx: int = 0

    @classmethod
    def initial(cls, n_champions: int) -> "DraftState":
        return cls(
            n_champions=n_champions,
            available=np.ones(n_champions, dtype=np.int8),
        )

    def clone(self) -> "DraftState":
        return DraftState(
            n_champions=self.n_champions,
            blue_picks=list(self.blue_picks),
            red_picks=list(self.red_picks),
            blue_bans=list(self.blue_bans),
            red_bans=list(self.red_bans),
            available=self.available.copy(),
            step_idx=self.step_idx,
        )

    def current_step(self) -> DraftStep | None:
        if self.step_idx >= len(DRAFT_SEQUENCE):
            return None
        return DRAFT_SEQUENCE[self.step_idx]

    def is_terminal(self) -> bool:
        return self.step_idx >= len(DRAFT_SEQUENCE)

    def legal_mask(self) -> np.ndarray:
        return self.available.astype(bool)

    def apply(self, action: int) -> Side:
        step = DRAFT_SEQUENCE[self.step_idx]
        if not (0 <= action < self.n_champions) or not self.available[action]:
            raise ValueError(f"Illegal action {action} at step {self.step_idx}.")
        if step.action_type == ActionType.BAN:
            (self.blue_bans if step.side == Side.BLUE else self.red_bans).append(action)
        else:
            (self.blue_picks if step.side == Side.BLUE else self.red_picks).append(
                action
            )
        self.available[action] = 0
        self.step_idx += 1
        return step.side

    def to_obs(self) -> dict[str, Any]:
        def _pad(xs: list[int]) -> np.ndarray:
            out = np.full(5, -1, dtype=np.int32)
            for i, v in enumerate(xs):
                out[i] = v
            return out

        step = self.current_step()
        return {
            "blue_picks": _pad(self.blue_picks),
            "red_picks": _pad(self.red_picks),
            "blue_bans": _pad(self.blue_bans),
            "red_bans": _pad(self.red_bans),
            "available_mask": self.available.copy(),
            "step": self.step_idx,
            "acting_side": 0 if step is None else int(step.side),
            "action_type": 0 if step is None else int(step.action_type),
        }

    def current_side(self) -> Side | None:
        step = self.current_step()
        return None if step is None else step.side
