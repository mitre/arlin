from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.ppo import PPO
from stable_baselines3.dqn import DQN

from typing import Type


def get_sb3_algo(algo_str: str) -> Type[BaseAlgorithm]:
    """
    Get the Algorithm class based on given algorithms string.

    Args:
        algo_str (str): Str representation of an algorithm class.

    Raises:
        ValueError: If given algorithm is not implemented.

    Returns:
        BaseAlgorithm: Algorithm class for the specified algorithm.
    """
    algo: Type[BaseAlgorithm]

    if algo_str == "dqn":
        algo = DQN
    elif algo_str == "ppo":
        algo = PPO
    else:
        raise ValueError(
            f"Error: Algorithm {algo_str} is not supported - has to be one of: [dqn, ppo]"
        )

    return algo
