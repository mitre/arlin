from abc import ABC, abstractmethod
import numpy as np

class BaseDatapointDict(ABC):
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
        
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    @abstractmethod
    def add_specific_datapoint(self, *kwargs):
        pass
    
    @abstractmethod
    def get_dict(self):
        data_dict = {
            "observations": np.as_array(self.observations),
            "actions": np.as_array(self.actions),
            "rewards": np.as_array(self.rewards),
            "dones": np.as_array(self.dones)
        }
        
        return data_dict
    

class PPODatapointDict(BaseDatapointDict):
    
    def __init__(self):
        super().__init__(self)
        self.latent_actors = []
        self.latent_critics = []
        self.dist_probs = []
        self.critic_values = []
    
    def add_specific_datapoint(
        self, 
        latent_actor,
        latent_critic,
        dist_prob,
        critic_value) -> None:
        
        self.latent_actors.append(latent_actor)
        self.latent_critics.append(latent_critic)
        self.dist_probs.append(dist_prob)
        self.critic_values.append(critic_value)
    
    def get_dict(self):
        data_dict = {
            "observations": np.as_array(self.observations),
            "actions": np.as_array(self.actions),
            "rewards": np.as_array(self.rewards),
            "dones": np.as_array(self.dones),
            "latent_actors": np.as_array(self.latent_actors),
            "latent_critics": np.as_array(self.latent_critics),
            "dist_probs": np.as_array(self.dist_probs),
            "critic_values": np.as_array(self.critic_values)
        }
        
        return data_dict
    
    
class DQNDatapointDict(BaseDatapointDict):
    
    def __init__(self):
        super().__init__(self)
        self.q_vals = []
        self.latent_qs = []
    
    def add_specific_datapoint(
        self, 
        q_val,
        latent_q) -> None:
        
        self.latent_actors.append(q_val)
        self.latent_critics.append(latent_q)
    
    def get_dict(self):
        data_dict = {
            "observations": np.as_array(self.observations),
            "actions": np.as_array(self.actions),
            "rewards": np.as_array(self.rewards),
            "dones": np.as_array(self.dones),
            "q_vals": np.as_array(self.q_vals),
            "latent_qs": np.as_array(self.latent_qs)
        }
        
        return data_dict