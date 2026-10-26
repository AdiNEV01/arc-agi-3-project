"""Shared observation and action encoding for ARC-AGI-3.

Used by BOTH the Gymnasium adapter (training/local evaluation) and the Kaggle
agent bridge, so the policy sees identical inputs in both settings.

Observation: one ARC response can contain several frames (animation steps).
We keep only the LAST frame, i.e. the settled state after the action.
It is placed top-left in a 64x64 uint8 grid. Cell values 0-15 are ARC colours;
PAD_VALUE (16) marks cells outside a smaller-than-64 frame.

Action map: a flat Discrete index over
  [ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7] + ACTION6 clicks
ACTION6 (x, y) is discretised into a click_grid x click_grid lattice of cell
centres. RESET is NOT exposed to the agent; the adapter/evaluator issues it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from arcengine import GameAction

GRID = 64
N_COLOURS = 16
PAD_VALUE = 16
SIMPLE_IDS = (1, 2, 3, 4, 5, 7)


def frame_to_grid(frames: Sequence[Any]) -> np.ndarray:
    """Convert the frame list from a response (ndarrays or nested lists) to a 64x64 grid."""
    grid = np.full((GRID, GRID), PAD_VALUE, dtype=np.uint8)
    if frames is None or len(frames) == 0:
        return grid
    last = np.asarray(frames[-1], dtype=np.int64)
    if last.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {last.shape}")
    h, w = min(last.shape[0], GRID), min(last.shape[1], GRID)
    grid[:h, :w] = np.clip(last[:h, :w], 0, N_COLOURS - 1)
    return grid


@dataclass(frozen=True)
class ActionSpec:
    action_id: int          # GameAction value (1-7)
    x: int | None = None    # only for ACTION6
    y: int | None = None

    @property
    def game_action(self) -> GameAction:
        return GameAction.from_id(self.action_id)

    @property
    def data(self) -> dict[str, int]:
        return {} if self.x is None else {"x": self.x, "y": self.y}

    def label(self) -> str:
        name = self.game_action.name
        return name if self.x is None else f"{name}({self.x},{self.y})"


class ActionMap:
    """Bijection between a flat Discrete index and ARC actions."""

    def __init__(self, click_grid: int = 8):
        if click_grid < 1 or GRID % click_grid:
            raise ValueError("click_grid must divide 64 (1, 2, 4, 8, 16, 32, 64)")
        self.click_grid = click_grid
        cell = GRID // click_grid
        specs = [ActionSpec(a) for a in SIMPLE_IDS]
        for row in range(click_grid):          # y
            for col in range(click_grid):      # x
                specs.append(ActionSpec(6, x=col * cell + cell // 2, y=row * cell + cell // 2))
        self.specs: list[ActionSpec] = specs

    def __len__(self) -> int:
        return len(self.specs)

    def decode(self, index: int) -> ActionSpec:
        if int(index) != index or not 0 <= int(index) < len(self.specs):
            raise ValueError(f"Invalid shared action index: {index}")
        return self.specs[int(index)]

    def mask(self, available_actions: Sequence[int]) -> np.ndarray:
        """Boolean mask of legal indices given the frame's available_actions (ints)."""
        avail = {int(a) for a in available_actions}
        return np.array([s.action_id in avail for s in self.specs], dtype=bool)

    def type_of(self, index: int) -> int:
        return self.specs[int(index)].action_id
