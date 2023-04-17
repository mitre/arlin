from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.ppo import PPO
from stable_baselines3.dqn import DQN

from arlin.dataset_creation.data_gatherer import BaseDataGatherer, PPODataGatherer, DQNDataGatherer
from arlin.dataset_creation.datapoint_dict import BaseDatapointDict, PPODatapointDict, DQNDatapointDict

from typing import Type


def get_algo(algo_str: str) -> Type[BaseAlgorithm]:
    """
    Get the Algorithm class based on given algorithms string.

    Args:
        algo_str (str): Str representation of an algorithm class.

    Raises:
        ValueError: If given algorithm is not implemented.

    Returns:
        BaseAlgorithm: Algorithm class for the specified algorithm.
    """
    algo: type[BaseAlgorithm]

    if algo_str == "dqn":
        algo = DQN
    elif algo_str == "ppo":
        algo = PPO
    else:
        raise ValueError(
            f"Error: Algorithm {algo_str} is not supported - has to be one of: [dqn, ppo]"
        )

    return algo

def get_datapoint_dict(algorithm: str) -> Type[BaseDatapointDict]:
    """
    Return DatapointDict based on given algorithm.
    
    Args:
        - algorithm (str): Algorithm used during training
        
    Returns:
        - Type[BaseDatapointDict]: Algorithm specific DatapointDict
    """
    
    if algorithm == "ppo":
        return PPODatapointDict
    elif algorithm == "dqn":
        return DQNDatapointDict
    else:
        raise ValueError("Unsupported algorithm given!")

def get_dataset_gatherer(algorithm: str) -> Type[BaseDataGatherer]:
    """
    Return the DataGatherer used for the given algorithm.
    
    Args:
        - algorithm (str): Algorithm used during training
    
    Returns:
        - Type[BaseDataGatherer]: Algorithm specific data gatherer
    """
    
    if algorithm == "ppo":
        return PPODataGatherer
    elif algorithm == "dqn":
        return DQNDataGatherer
    else:
        raise ValueError("Unsupported algorithm given!")