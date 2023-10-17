import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class BaseDatapoint:
    """Base datapoint with traditional RL data that is common to all algorithms."""

    observations: Optional[np.ndarray] = None
    actions: Optional[int] = None
    rewards: Optional[float] = None
    terminateds: Optional[bool] = None
    truncateds: Optional[bool] = None
    steps: Optional[float] = None
    renders: Optional[np.ndarray] = None

    def __eq__(self, other: Any):
        if not isinstance(other, BaseDatapoint):
            return False

        self_fields = [i.name for i in dataclasses.fields(self)]
        other_fields = [i.name for i in dataclasses.fields(other)]

        if not self_fields == other_fields:
            return False

        for field in self_fields:
            if not np.array_equal(getattr(self, field), getattr(other, field)):
                return False

        return True

    def add_base_data(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        step: float,
        render: np.ndarray,
    ):
        """Add the base RL data to this Datapoint object.

        Args:
            obs (np.ndarray): Current observation
            action (int): Action taken
            reward (float): Reward received
            terminated (bool): Did the episode end
            truncated (bool): Did we run out of steps
            step (float): Current step of this data
            render (np.ndarray): Render of the environment state
        """
        self.observations = obs
        self.actions = action
        self.rewards = reward
        self.terminateds = terminated
        self.truncateds = truncated
        self.steps = step
        self.renders = render


@dataclass
class SB3PPODatapoint(BaseDatapoint):
    """Datapoint for a PPO algorithm trained in stable-baselines3."""

    latent_actors: Optional[np.ndarray] = None
    latent_critics: Optional[np.ndarray] = None
    dist_probs: Optional[np.ndarray] = None
    critic_values: Optional[float] = None
    pi_features: Optional[np.ndarray] = None
    vf_features: Optional[np.ndarray] = None


@dataclass
class SB3DQNDatapoint(BaseDatapoint):
    """Datapoint for a DQN algorithm trained in stable-baselines3."""

    q_vals: Optional[np.ndarray] = None
    latent_qs: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
