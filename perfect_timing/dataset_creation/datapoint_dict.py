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
        self.dones = []
    
    def add_base_data(
        self, 
        observation: np.ndarray, 
        action: int, 
        reward: float, 
        done: bool
        ) -> None:
        """
        Add base data that is independent of the algorithm used during training.
        
        Args:
            - observation (np.ndarray): Observation that was passed through the model
            - action (int): Output action from model
            - reward (float): Output reward from given step
            - done (bool): Whether or not this was the last step in an episode
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
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
            "observations": np.as_array(self.observations),
            "actions": np.as_array(self.actions),
            "rewards": np.as_array(self.rewards),
            "dones": np.as_array(self.dones)
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
    
    def add_specific_datapoint(
        self, 
        latent_actor: np.ndarray,
        latent_critic: np.ndarray,
        dist_prob: np.ndarray,
        critic_value: float) -> None:
        """
        Add datapoint information that is specific to PPO.
        
        Args:
            - latent_actor (np.ndarray): Final embedding layer output within actor
            - latent_critic (np.ndarray): Final embedding layer output within critic
            - dist_probs (np.ndarray): Probability distribution of actions
            - critic_value (float): Critic value for the current state
        """
        
        self.latent_actors.append(latent_actor)
        self.latent_critics.append(latent_critic)
        self.dist_probs.append(dist_prob)
        self.critic_values.append(critic_value)
    
    def get_dict(self) -> Dict[str, np.ndarray]:
        data_dict = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "latent_actors": np.array(self.latent_actors),
            "latent_critics": np.array(self.latent_critics),
            "dist_probs": np.array(self.dist_probs),
            "critic_values": np.array(self.critic_values)
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
    
    def add_specific_datapoint(
        self, 
        q_vals,
        latent_q) -> None:
        """
        Add datapoint information that is specific to DQN.
        
        Args:
            - q_vals (np.ndarray): Q values for each action at current state
            - latent_q (np.ndarray): Final embedding layer output within q network
        """
        
        self.q_vals.append(q_vals)
        self.latent_qs.append(latent_q)
    
    def get_dict(self) -> Dict[str, np.ndarray]:
        data_dict = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "q_vals": np.array(self.q_vals),
            "latent_qs": np.array(self.latent_qs)
        }
        
        return data_dict