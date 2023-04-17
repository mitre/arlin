from stable_baselines3.common.policies import BasePolicy
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Type
import torch as th

from arlin.dataset_creation.datapoint_dict import BaseDatapointDict, PPODatapointDict, DQNDatapointDict

class BaseDataGatherer(ABC):
    """
    #### Description
    Base class for gathering XRL data from a policy.
    
    Args:
        - policy (BasePolicy): Policy from which to extract XRL data from.
    """
    def __init__(self, policy: BasePolicy):
        self.policy = policy
    
    @abstractmethod
    def gather_data(self, 
                    observation: np.ndarray, 
                    datapoint_dict: Type[BaseDatapointDict]) -> int:
        """
        Gather the data from one step and insert it into the given dataset_dict.
        
        Args:
            - observation (np.ndarray): Current observation from the environment
            - datapoint_dict (Type[BaseDatapointDict]): Datapoint dictionary of all values
            
        Returns:
            - int: Action to take
        """
        pass


class PPODataGatherer(BaseDataGatherer):
    
    def __init__(self, policy: Type[BasePolicy]):
        super().__init__(policy)
        
    def gather_data(self, 
                    observation: np.ndarray, 
                    datapoint_dict: PPODatapointDict) -> int:
        
        with th.no_grad():
            obs = th.Tensor(np.expand_dims(observation, 0))
            policy_dist = self.policy.get_distribution(obs)
            action = policy_dist.get_actions(deterministic=True).item()
            probs = policy_dist.distribution.probs
            value = self.policy.predict_values(obs)
            
            features = self.policy.extract_features(obs)
            if self.policy.share_features_extractor:
                latent_pi, latent_vf = self.policy.mlp_extractor(features)
                pi_features = features
                vf_features = features
            else:
                pi_features, vf_features = features
                latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)
        
        datapoint_dict.add_specific_datapoint(th.squeeze(latent_pi).numpy(), 
                                              th.squeeze(latent_vf).numpy(), 
                                              th.squeeze(probs).numpy(), 
                                              th.squeeze(value).item(),
                                              th.squeeze(pi_features).numpy(),
                                              th.squeeze(vf_features).numpy())
        
        return action

class DQNDataGatherer(BaseDataGatherer):
    
    def __init__(self, policy: Type[BasePolicy]):
        super().__init__(policy)
        
    def gather_data(self, 
                    observation: np.ndarray, 
                    datapoint_dict: DQNDatapointDict) -> int:
        
        with th.no_grad():
            obs = th.Tensor(np.expand_dims(observation, 0))
            
            features = self.policy.extract_features(obs, self.policy.q_net.features_extractor)
            latent_q = self.policy.q_net.q_net[:-1](features)
            q_vals = self.policy.q_net.q_net[-1](latent_q)
            action = q_vals.argmax(dim=1).reshape(-1).item()
        
        datapoint_dict.add_specific_datapoint(th.squeeze(q_vals).numpy(), 
                                              th.squeeze(latent_q).numpy(),
                                              th.squeeze(features).numpy())
        
        return action