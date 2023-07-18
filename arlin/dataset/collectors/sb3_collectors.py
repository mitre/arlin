import numpy as np
import torch as th
from typing import Tuple, Type

from arlin.dataset.collectors import BaseDataCollector, BaseDatapoint
from stable_baselines3.common.base_class import BasePolicy

class SB3PPODataCollector(BaseDataCollector):
    
    def __init__(self, 
                 datapoint_cls: Type[BaseDatapoint],
                 policy: BasePolicy
                 ):
        super().__init__(datapoint_cls=datapoint_cls)
        self.policy = policy
        
    def collect_internal_data(self, 
                              observation: np.ndarray) -> Tuple[type[BaseDatapoint], int]:
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
        
        datapoint = self.datapoint_cls(latent_actors = th.squeeze(latent_pi).numpy(), 
                                        latent_critics = th.squeeze(latent_vf).numpy(), 
                                        dist_probs = th.squeeze(probs).numpy(), 
                                        critic_values = th.squeeze(value).item(),
                                        pi_features = th.squeeze(pi_features).numpy(),
                                        vf_features = th.squeeze(vf_features).numpy())
        
        return datapoint, action

class SB3DQNDataCollector(BaseDataCollector):
    
    def __init__(self, 
                 datapoint_cls: Type[BaseDatapoint],
                 policy: BasePolicy
                 ):
        super().__init__(datapoint_cls=datapoint_cls)
        self.policy = policy
        
    def collect_internal_data(self,
                              observation: np.ndarray) -> Tuple[type[BaseDatapoint], int]:
        with th.no_grad():
            obs = th.Tensor(np.expand_dims(observation, 0))
            
            features = self.policy.extract_features(obs, self.policy.q_net.features_extractor)
            latent_q = self.policy.q_net.q_net[:-1](features)
            q_vals = self.policy.q_net.q_net[-1](latent_q)
            action = q_vals.argmax(dim=1).reshape(-1).item()
        
        datapoint = self.datapoint_cls(q_vals = th.squeeze(q_vals).numpy(),
                                        latent_q = th.squeeze(latent_q).numpy(),
                                        features = th.squeeze(features).numpy())
        
        return datapoint, action


