import pytest
from stable_baselines3.dqn import DQN
from stable_baselines3.ppo import PPO

from arlin.utils.sb3_utils import get_sb3_algo


class TestSB3Utils:
    def test_get_sb3_algo(self):
        ppo_algo_0 = get_sb3_algo("ppo")
        ppo_algo_1 = get_sb3_algo("PPO")
        dqn_algo = get_sb3_algo("dqn")

        assert ppo_algo_0 == PPO
        assert ppo_algo_1 == PPO
        assert dqn_algo == DQN

        with pytest.raises(ValueError):
            get_sb3_algo("a2c")
