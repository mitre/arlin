from abc import ABC, abstractmethod
from typing import Tuple, Type
import gymnasium as gym
import numpy as np

from arlin.dataset.collectors import BaseDatapoint
from arlin.dataset.collectors.datapoints import *

class BaseDataCollector(ABC):
    def __init__(self, 
                 datapoint_cls: Type[BaseDatapoint]):
        self.datapoint_cls = datapoint_cls
    
    @abstractmethod
    def collect_internal_data(self, 
                              observation: np.ndarray) -> Tuple[type[BaseDatapoint], int]:
        pass

class RandomDataCollector(BaseDataCollector):
    
    def __init__(self,
                 datapoint_cls: Type[BaseDatapoint],
                 environment: gym.Env):
        super().__init__(datapoint_cls=datapoint_cls)
        self.env = environment
        
    def collect_internal_data(self,
                              observation: np.ndarray) -> Tuple[type[BaseDatapoint], int]:
        action = self.env.action_space.sample()
        return self.datapoint_cls(), action
