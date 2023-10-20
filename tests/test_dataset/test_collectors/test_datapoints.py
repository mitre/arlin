import gymnasium as gym
import pytest

from arlin.dataset.collectors import RandomDataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint


@pytest.fixture
def env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


class TestBaseDatapoint:
    def test_add_base_data(self, env):
        collector = RandomDataCollector(BaseDatapoint, env)

        obs, _ = env.reset()
        datapoint, action = collector.collect_internal_data(obs)

        assert isinstance(datapoint, BaseDatapoint)
        assert datapoint.observations is None
        assert datapoint.actions is None
        assert datapoint.rewards is None
        assert datapoint.terminateds is None
        assert datapoint.truncateds is None
        assert datapoint.steps is None
        assert datapoint.renders is None

        obs, reward, terminated, truncated, _ = env.step(action)
        render = env.render()
        datapoint.add_base_data(obs, action, reward, terminated, truncated, 0, render)

        assert datapoint.observations is not None
        assert datapoint.actions is not None
        assert datapoint.rewards is not None
        assert datapoint.terminateds is not None
        assert datapoint.truncateds is not None
        assert datapoint.steps is not None
        assert datapoint.renders is not None
