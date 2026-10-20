"""Optional policy feature extractors for the PPO experiments."""

from __future__ import annotations

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class GridCNNFeaturesExtractor(BaseFeaturesExtractor):
    """A small CNN that preserves the local structure of grid-like images.

    The default Stable-Baselines3 CNN is retained for the baseline. This
    extractor is an experimental representation improvement: it uses small
    stride-1 filters and pooling so small puzzle elements are not immediately
    downsampled by a large first convolution.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        if len(observation_space.shape) != 3:
            raise ValueError(
                "GridCNNFeaturesExtractor expects channels-first images: "
                "(channels, height, width)."
            )

        channels = observation_space.shape[0]
        super().__init__(observation_space, features_dim)

        self.convolution = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64 * 4 * 4, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(observations))
