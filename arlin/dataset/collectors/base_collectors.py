from abc import ABC, abstractmethod
from typing import Tuple, Type

import gymnasium as gym
import numpy as np

from arlin.dataset.collectors import BaseDatapoint


class BaseDataCollector(ABC):
    """Base class for data collection."""

    def __init__(self, datapoint_cls: Type[BaseDatapoint]):
        """Initialize a BaseDataCollector object.

        Args:
            datapoint_cls (Type[BaseDatapoint]): Class of datapoint we are collecting.
        """
        self.datapoint_cls = datapoint_cls

    @abstractmethod
    def collect_internal_data(
        self, observation: np.ndarray
    ) -> Tuple[Type[BaseDatapoint], int]:
        """Collect internal model-specific data.

        Args:
            observation (np.ndarray): Input observation to the model

        Returns:
            Tuple[Type[BaseDatapoint], int]: Internal data and action to take
        """
        pass


class RandomDataCollector(BaseDataCollector):
    """Data collection when the agent is taking random actions."""

    def __init__(self, datapoint_cls: Type[BaseDatapoint], environment: gym.Env):
        """Initialize a RandomDataCollector object.

        Args:
            datapoint_cls (Type[BaseDatapoint]): Class of datapoint we are collecting.
            environment (gym.Env): Environment the policy is interacting with.
        """
        super().__init__(datapoint_cls=datapoint_cls)
        self.env = environment

    def collect_internal_data(
        self, observation: np.ndarray
    ) -> Tuple[Type[BaseDatapoint], int]:
        action = self.env.action_space.sample()
        return self.datapoint_cls(), action
