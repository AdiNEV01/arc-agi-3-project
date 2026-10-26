
"""Baseline CNN-based Deep Q-Network for ARC-AGI-3."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DQN.action_codec import (
    GRID_SIZE,
    N_SIMPLE,
    N_ACTIONS,
)

NUM_COLORS = 16


def observation_to_tensor(obs, device="cpu"):
    """Convert the latest ARC frame into one-hot CNN input."""
    if isinstance(obs, np.ndarray):
        return encode_frames(obs, device=device)
    if not obs.frame:
        raise ValueError("Observation contains no frames")

    frame = np.asarray(obs.frame[-1]) # Convert to numpy array for processing

    if frame.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"Expected 64x64 frame, got {frame.shape}"
        )

    if not np.issubdtype(frame.dtype, np.integer):
        raise ValueError("Expected integer color indices")

    if frame.min() < 0 or frame.max() >= NUM_COLORS:
        raise ValueError("Color index outside 0-15")

    indices = torch.as_tensor(frame.astype(np.int64), device=device) # Convert to tensor

    # (64,64) -> (16,64,64) -> (1,16,64,64)
    encoded = F.one_hot(
        indices,
        num_classes=NUM_COLORS
    ).permute(2, 0, 1).float() # Convert to float for CNN input

    return encoded.unsqueeze(0)


def encode_frames(frames, device="cpu"):
    """[F,H,W] or [B,F,H,W] -> [B,F*16,H,W], frame-major channels."""
    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4 or frames.shape[-2:] != (64, 64):
        raise ValueError(f"Unexpected frame history shape: {frames.shape}")
    if not np.issubdtype(frames.dtype, np.integer) or frames.min() < 0 or frames.max() > 16:
        raise ValueError("Expected integer colors 0-15 or shared padding 16")
    # Transfer compact replay grids; expand to int64 and one-hot on the device.
    indices = torch.as_tensor(frames.astype(np.uint8, copy=False), device=device).long()
    b, f, h, w = indices.shape
    return F.one_hot(indices, 17)[..., :16].permute(0, 1, 4, 2, 3).reshape(b, f * 16, h, w).float()


class QNetwork(nn.Module):
    def __init__(self, num_frames=1):
        super().__init__()
        self.num_frames = num_frames

        # Shared visual feature extractor.
        self.encoder = nn.Sequential(
            nn.Conv2d(16 * num_frames, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Q-values for ordinary discrete actions.
        self.discrete_head = nn.Sequential(
            # The encoder is always 64x64. These non-overlapping 16x16
            # means equal adaptive 64->4 pooling, whose CUDA backward
            # is unsupported by strict deterministic PyTorch execution.
            # Both pooling layers have no state-dict parameters.
            nn.AvgPool2d(kernel_size=16, stride=16),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, N_SIMPLE),
        )

        # Q-values for all 64x64 click positions.
        self.spatial_head = nn.Conv2d(
            64, 1, kernel_size=1
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (16 * self.num_frames, 64, 64):
            raise ValueError(
                f"Expected [B,{16 * self.num_frames},64,64], got {tuple(x.shape)}"
            )

        features = self.encoder(x)

        discrete_q = self.discrete_head(features)

        spatial_q = self.spatial_head(features)
        spatial_q = spatial_q.flatten(start_dim=1)

        q_values = torch.cat(
            [discrete_q, spatial_q],
            dim=1
        )

        assert q_values.shape[1] == N_ACTIONS

        return q_values
