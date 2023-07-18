from dataclasses import dataclass
import numpy as np

from typing import Optional

@dataclass
class BaseDatapoint():
    observations: Optional[np.ndarray] = None
    actions: Optional[int] = None
    rewards: Optional[float] = None
    dones: Optional[bool] = None
    steps: Optional[int] = None
    
    def add_base_data(self,
                      obs: np.ndarray,
                      action: int,
                      reward: float,
                      done: bool,
                      step: int):
        
        self.observations = obs
        self.actions = action
        self.rewards = reward
        self.dones = done
        self.steps = step

@dataclass
class SB3PPODatapoint(BaseDatapoint):
    latent_actors: Optional[np.ndarray] = None
    latent_critics: Optional[np.ndarray] = None
    dist_probs: Optional[np.ndarray] = None
    critic_values: Optional[float] = None
    pi_features: Optional[np.ndarray] = None
    vf_features: Optional[np.ndarray] = None

@dataclass
class SB3DQNDatapoint(BaseDatapoint):
    q_vals: Optional[np.ndarray] = None
    latent_qs: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None