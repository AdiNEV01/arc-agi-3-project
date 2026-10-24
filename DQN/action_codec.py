
"""Encode and decode ARC-AGI-3 actions for DQN."""

from arcengine import GameAction
import torch

GRID_SIZE = 64

# All simple game actions, excluding RESET.
# RESET is handled by the episode controller, not DQN.
SIMPLE_ACTIONS = tuple(
    action for action in GameAction
    if action is not GameAction.RESET
    and not action.is_complex()
) # simple actions are those that do not require additional data (like coordinates)

N_SIMPLE = len(SIMPLE_ACTIONS)
N_SPATIAL = GRID_SIZE * GRID_SIZE
N_ACTIONS = N_SIMPLE + N_SPATIAL


def valid_action_mask(available_actions):
    """Return a Boolean mask of legal DQN action indices."""
    available = set(available_actions)

    unsupported = [
        action for action in available
        if action.is_complex()
        and action is not GameAction.ACTION6
    ]
    if unsupported:
        raise NotImplementedError(
            f"Unsupported complex actions: {unsupported}"
        )

    mask = torch.zeros(N_ACTIONS, dtype=torch.bool)

    # Enable permitted simple actions.
    for index, action in enumerate(SIMPLE_ACTIONS):
        if action in available:
            mask[index] = True

    # Enable coordinate clicks when ACTION6 is available.
    if GameAction.ACTION6 in available:
        mask[N_SIMPLE:] = True

    if not mask.any().item():
        raise ValueError("No supported actions available")

    return mask


def decode_action(index):
    """Convert a DQN action index into an ARC action and data."""
    index = int(index)

    if not 0 <= index < N_ACTIONS:
        raise ValueError(f"Invalid action index: {index}")

    if index < N_SIMPLE:
        return SIMPLE_ACTIONS[index], None

    position = index - N_SIMPLE

    x = position % GRID_SIZE
    y = position // GRID_SIZE

    return GameAction.ACTION6, {"x": x, "y": y}


def encode_action(action, data=None):
    """Convert an ARC action into its DQN action index."""
    if action in SIMPLE_ACTIONS:
        return SIMPLE_ACTIONS.index(action)

    if action is GameAction.ACTION6:
        if data is None:
            raise ValueError("ACTION6 requires x,y coordinates")

        x, y = int(data["x"]), int(data["y"])

        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            raise ValueError("Coordinates outside 64x64 grid")

        return N_SIMPLE + y * GRID_SIZE + x

    raise ValueError(f"Unsupported DQN action: {action}")
