# utils/game_flow.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

POS_9MAX = ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"]

@dataclass
class GameFlow:
    positions: List[str] = field(default_factory=lambda: POS_9MAX.copy())
    active: Dict[str, bool] = field(default_factory=lambda: {p: True for p in POS_9MAX})

    def fold(self, pos: str) -> None:
        self.active[pos] = False

    def is_active(self, pos: str) -> bool:
        return self.active.get(pos, False)

    def active_positions(self) -> List[str]:
        return [p for p in self.positions if self.is_active(p)]

    def idx(self, pos: str) -> int:
        return self.positions.index(pos)

    def clockwise_between(self, a: str, b: str) -> List[str]:
        """Strictly between a -> b clockwise, wrapping once."""
        n = len(self.positions)
        i, j = self.idx(a), self.idx(b)
        out = []
        k = (i + 1) % n
        while k != j:
            out.append(self.positions[k])
            k = (k + 1) % n
        return out

    def set_action_start(self, start_pos: str) -> None:
        """If action starts at UTG2, UTG & UTG1 are considered already folded."""
        for p in self.clockwise_between(start_pos, start_pos):  # all other seats
            # fold those that are BEFORE start_pos in list ordering (preflop opener)
            if self.idx(p) < self.idx(start_pos):
                self.fold(p)
