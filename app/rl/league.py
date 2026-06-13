"""Adversarial opponent league: PFSP sampling + SPRT promotion gating.

Frozen-checkpoint pool for AlphaStar-style prioritised fictitious self-play and
Stockfish-fishtest-style SPRT promotion. Torch-free: checkpoint payloads are
opaque bytes supplied by the trainer.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LeagueEntry:
    path: str
    rating: float = 0.0
    games: int = 0   # learner games played vs this entry
    wins: int = 0    # learner wins vs this entry


@dataclass
class League:
    entries: list[LeagueEntry] = field(default_factory=list)
    champion_idx: int = 0

    @property
    def champion(self) -> LeagueEntry | None:
        """Return current champion or None if no entries."""
        if self.entries:
            return self.entries[self.champion_idx]
        return None

    def sample_opponent(self, rng: np.random.Generator, p: float = 2.0) -> tuple[int, LeagueEntry]:
        """PFSP: sample opponent weighted by (1 - learner_winrate) ^ p.

        Higher weight = learner loses more = harder opponent.
        """
        weights = []
        for entry in self.entries:
            if entry.games > 0:
                wr = entry.wins / entry.games
            else:
                wr = 0.5
            weight = (1.0 - wr) ** p
            weights.append(weight)

        w = np.asarray(weights, float)
        if w.sum() <= 0:
            probs = np.ones(len(self.entries)) / len(self.entries)
        else:
            probs = w / w.sum()

        idx = int(rng.choice(len(self.entries), p=probs))
        return idx, self.entries[idx]

    def record(self, idx: int, agent_won: bool) -> None:
        """Record game result against entry at idx."""
        self.entries[idx].games += 1
        if agent_won:
            self.entries[idx].wins += 1

    def admit(self, state_bytes: bytes, rating: float, directory: str | Path) -> LeagueEntry:
        """Admit new entry: write checkpoint and register in league."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"entry_{len(self.entries)}.pt"
        path.write_bytes(state_bytes)
        entry = LeagueEntry(path=str(path), rating=float(rating))
        self.entries.append(entry)
        self.champion_idx = len(self.entries) - 1
        return entry

    def save(self, directory: str | Path) -> None:
        """Persist league state to index.json."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "champion_idx": self.champion_idx,
            "entries": [asdict(e) for e in self.entries],
        }
        (directory / "index.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> League:
        """Load league state from index.json or return empty league."""
        directory = Path(directory)
        index_path = directory / "index.json"
        if not index_path.exists():
            return cls()
        data = json.loads(index_path.read_text())
        entries = [LeagueEntry(**e) for e in data["entries"]]
        champion_idx = int(data.get("champion_idx", 0))
        return cls(entries=entries, champion_idx=champion_idx)


def _elo_to_p(elo: float) -> float:
    """Logistic Elo -> expected score in (0,1)."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt(
    wins: int, losses: int, draws: int = 0, *,
    elo0: float = 0.0, elo1: float = 15.0,
    alpha: float = 0.05, beta: float = 0.05,
) -> str:
    """Wald SPRT (binary Elo model). Returns 'accept' | 'reject' | 'continue'.

    H0: learner Elo <= elo0 vs H1: >= elo1. Draws folded as half win + half loss.
    """
    p0, p1 = _elo_to_p(elo0), _elo_to_p(elo1)
    if not (0 < p0 < 1 and 0 < p1 < 1) or p0 == p1:
        return "continue"
    w = wins + 0.5 * draws
    l = losses + 0.5 * draws
    llr = w * math.log(p1 / p0) + l * math.log((1.0 - p1) / (1.0 - p0))
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    if llr >= upper:
        return "accept"
    if llr <= lower:
        return "reject"
    return "continue"


def elo_update(r_a: float, r_b: float, score_a: float, k: float = 16.0) -> tuple[float, float]:
    """Standard Elo update; total rating is conserved. score_a in [0,1]."""
    e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    return r_a + k * (score_a - e_a), r_b + k * ((1.0 - score_a) - (1.0 - e_a))
