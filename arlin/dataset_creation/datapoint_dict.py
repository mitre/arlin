from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class BaseDatapointDict(ABC):
    """
    #### Description
    Base class for storing XRL datapoints gathered from a trained policy.
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.total_rewards = []
        self.dones = []
        self.steps = []
    
    def add_base_data(
        self, 
        observation: np.ndarray, 
        action: int, 
        reward: float, 
        total_reward: float,
        done: bool,
        step: int
        ) -> None:
        """
        Add base data that is independent of the algorithm used during training.
        
        Args:
            - observation (np.ndarray): Observation that was passed through the model
            - action (int): Output action from model
            - reward (float): Output reward from given step
            - done (bool): Whether or not this was the last step in an episode
            - step (int): Current step
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_rewards.append(total_reward)
        self.dones.append(done)
        self.steps.append(step)
    
    @abstractmethod
    def add_specific_datapoint(self, *kwargs):
        """
        Add datapoint information that is specific to the algorithm used during training.
        """
        pass
    
    @abstractmethod
    def get_dict(self) -> Dict[str, np.ndarray]:
        """
        Return a dicitonary representation of all datapoints currently collected.
        """
        data_dict = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "total_rewards": np.array(self.total_rewards),
            "dones": np.array(self.dones),
            "steps": np.array(self.steps)
        }
        
        return data_dict
    

class PPODatapointDict(BaseDatapointDict):
    """
    #### Datapoint dictionary for storing XRL datapoints from a PPO algorithm
    """
    
    def __init__(self):
        super().__init__()
        self.latent_actors = []
        self.latent_critics = []
        self.dist_probs = []
        self.critic_values = []
        self.pi_features = []
        self.vf_features = []
    
    def add_specific_datapoint(
        self, 
        latent_actor: np.ndarray,
        latent_critic: np.ndarray,
        dist_prob: np.ndarray,
        critic_value: float,
        pi_features: np.ndarray,
        vf_features: np.ndarray) -> None:
        """
        Add datapoint information that is specific to PPO.
        
        Args:
            - latent_actor (np.ndarray): Final embedding layer output within actor
            - latent_critic (np.ndarray): Final embedding layer output within critic
            - dist_probs (np.ndarray): Probability distribution of actions
            - critic_value (float): Critic value for the current state
            - pi_features (np.ndarray): Output features for the obs from the pi net
            - vf_features (np.ndarray): Output features for the obs from the vf net
        """
        
        self.latent_actors.append(latent_actor)
        self.latent_critics.append(latent_critic)
        self.dist_probs.append(dist_prob)
        self.critic_values.append(critic_value)
        self.pi_features.append(pi_features)
        self.vf_features.append(vf_features)
    
    def get_dict(self) -> Dict[str, np.ndarray]:
        data_dict = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "total_rewards": np.array(self.total_rewards),
            "dones": np.array(self.dones),
            "steps": np.array(self.steps),
            "latent_actors": np.array(self.latent_actors),
            "latent_critics": np.array(self.latent_critics),
            "dist_probs": np.array(self.dist_probs),
            "critic_values": np.array(self.critic_values),
            "pi_features": np.array(self.pi_features),
            "vf_features": np.array(self.vf_features)
        }
        
        return data_dict
    
    
class DQNDatapointDict(BaseDatapointDict):
    """
    #### Datapoint dictionary for storing XRL datapoints from a DQN algorithm
    """
    def __init__(self):
        super().__init__()
        self.q_vals = []
        self.latent_qs = []
        self.features = []
    
    def add_specific_datapoint(
        self, 
        q_vals: np.ndarray,
        latent_q: np.ndarray,
        features: np.ndarray) -> None:
        """
        Add datapoint information that is specific to DQN.
        
        Args:
            - q_vals (np.ndarray): Q values for each action at current state
            - latent_q (np.ndarray): Final embedding layer output within q network
            - features (np.ndarray): Output features for the obs
        """
        
        self.q_vals.append(q_vals)
        self.latent_qs.append(latent_q)
        self.features.append(features)
    
    def get_dict(self) -> Dict[str, np.ndarray]:
        data_dict = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "total_rewards": np.array(self.total_rewards),
            "dones": np.array(self.dones),
            "steps": np.as_array(self.steps),
            "q_vals": np.array(self.q_vals),
            "latent_qs": np.array(self.latent_qs),
            "features": np.array(self.features)
        }
        
        return data_dict