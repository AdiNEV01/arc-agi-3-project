
"""Baseline DQN learning agent for ARC-AGI-3."""

import random

import numpy as np
import torch
import torch.nn.functional as F

from DQN.action_codec import (
    N_ACTIONS,
    valid_action_mask,
)
from DQN.q_network import (
    QNetwork,
    observation_to_tensor,
    encode_frames,
)
from DQN.replay_buffer import ReplayBuffer


def extract_frame(obs):
    """Extract and validate an ARC observation grid."""
    frame = np.asarray(obs if isinstance(obs, np.ndarray) else obs.frame[-1])

    if frame.shape[-2:] != (64, 64):
        raise ValueError(
            f"Unexpected frame shape: {frame.shape}"
        )

    if frame.min() < 0 or frame.max() >= 16:
        raise ValueError(
            "Observation has values outside 0-15"
        )

    return frame.astype(np.uint8, copy=True)


class DQNAgent:
    def __init__(
        self,
        seed=42,
        lr=1e-4,
        gamma=0.99,
        batch_size=8,
        buffer_capacity=3000,
        target_update_interval=250,
        device="cpu",
        num_frames=1,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = (
            target_update_interval
        )

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # Online Q-network: updated by gradient descent.
        self.online = QNetwork(num_frames=num_frames).to(self.device)

        # Target Q-network: periodically synchronised.
        self.target = QNetwork(num_frames=num_frames).to(self.device)
        self.target.load_state_dict(
            self.online.state_dict()
        )
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=lr,
        )

        self.replay = ReplayBuffer(
            capacity=buffer_capacity
        )

        self.gradient_steps = 0

    def select_action(
        self, obs, available_actions, epsilon
    ):
        """Choose an action index using epsilon-greedy."""
        mask = valid_action_mask(available_actions)

        if self.rng.random() < epsilon:
            candidates = torch.where(mask)[0].tolist()
            return self.rng.choice(candidates)

        with torch.no_grad():
            state = observation_to_tensor(
                obs, device=self.device
            )
            q_values = self.online(state)[0]

            legal_q = q_values.masked_fill(
                ~mask.to(self.device),
                float("-inf"),
            )

            return int(legal_q.argmax().item())

    def remember(
        self,
        obs,
        action_index,
        reward,
        next_obs,
        terminated,
        next_available_actions,
    ):
        """Store one transition and next-state action mask."""
        if terminated:
            next_mask = np.zeros(
                N_ACTIONS, dtype=np.bool_
            )
        else:
            next_mask = valid_action_mask(
                next_available_actions
            ).numpy()

        self.replay.add(
            state=extract_frame(obs),
            action=action_index,
            reward=reward,
            next_state=extract_frame(next_obs),
            terminated=terminated,
            next_valid_mask=next_mask,
        )

    def encode_batch(self, frames):
        """Convert [B,64,64] grids to [B,16,64,64]."""
        stacked = np.stack(frames)
        if stacked.ndim == 4:
            return encode_frames(stacked, device=self.device)
        indices = torch.as_tensor(
            stacked,
            dtype=torch.long,
            device=self.device,
        )

        return F.one_hot(
            indices, num_classes=16
        ).permute(0, 3, 1, 2).float()

    def learn(self):
        """Perform one DQN gradient update."""
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(
            self.batch_size, self.rng
        )

        states = self.encode_batch(
            [t.state for t in batch]
        )
        next_states = self.encode_batch(
            [t.next_state for t in batch]
        )

        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        terminated = torch.tensor(
            [t.terminated for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        next_masks = torch.as_tensor(
            np.stack([
                t.next_valid_mask for t in batch
            ]),
            dtype=torch.bool,
            device=self.device,
        )

        # Q(s, a) from the trainable online network.
        predicted_q = self.online(
            states
        ).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Bellman target from the frozen target network.
        with torch.no_grad():
            next_q = self.target(next_states)

            next_q = next_q.masked_fill(
                ~next_masks, -1e9
            )

            next_max_q = next_q.max(dim=1).values

            # Terminal states have no future Q-value.
            has_valid_next = next_masks.any(dim=1)

            next_max_q = torch.where(
                has_valid_next,
                next_max_q,
                torch.zeros_like(next_max_q),
            )

            targets = (
                rewards
                + self.gamma
                * (1.0 - terminated)
                * next_max_q
            )

        # Huber loss is commonly used in DQN.
        loss = F.smooth_l1_loss(
            predicted_q, targets
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()
        self.gradient_steps += 1

        if (
            self.gradient_steps
            % self.target_update_interval == 0
        ):
            self.target.load_state_dict(
                self.online.state_dict()
            )

        return float(loss.item())
