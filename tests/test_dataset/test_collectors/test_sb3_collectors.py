import gymnasium as gym
import pytest

from arlin.dataset.collectors import SB3DQNDataCollector, SB3PPODataCollector
from arlin.dataset.collectors.datapoints import SB3DQNDatapoint, SB3PPODatapoint
from arlin.dataset.loaders.sb3_loaders import load_hf_sb_model


@pytest.fixture
def ppo_policy():
    model = load_hf_sb_model(
        repo_id="sb3/ppo-CartPole-v1", filename="ppo-CartPole-v1.zip", algo_str="ppo"
    )

    return model.policy


@pytest.fixture
def dqn_policy():
    model = load_hf_sb_model(
        repo_id="sb3/dqn-CartPole-v1", filename="dqn-CartPole-v1.zip", algo_str="dqn"
    )

    return model.policy


@pytest.fixture
def env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


class TestSB3Collectors:
    def test_sb3_ppo_collector(self, ppo_policy, env):
        collector = SB3PPODataCollector(SB3PPODatapoint, ppo_policy)

        assert collector.datapoint_cls == SB3PPODatapoint

        obs, _ = env.reset()
        datapoint, action = collector.collect_internal_data(obs)

        assert isinstance(datapoint, SB3PPODatapoint)
        assert datapoint.observations is None
        assert datapoint.actions is None
        assert datapoint.rewards is None
        assert datapoint.terminateds is None
        assert datapoint.truncateds is None
        assert datapoint.steps is None
        assert datapoint.renders is None
        assert datapoint.latent_actors is not None
        assert datapoint.latent_critics is not None
        assert datapoint.dist_probs is not None
        assert datapoint.critic_values is not None
        assert datapoint.pi_features is not None
        assert datapoint.vf_features is not None
        assert env.action_space.contains(action)

        new_obs, _, _, _, _ = env.step(action)
        _, action = collector.collect_internal_data(new_obs)
        assert env.action_space.contains(action)

        new_obs, _, _, _, _ = env.step(action)
        _, action = collector.collect_internal_data(new_obs)
        assert env.action_space.contains(action)

        new_obs, _, _, _, _ = env.step(action)
        _, action = collector.collect_internal_data(new_obs)
        assert env.action_space.contains(action)

    def test_sb3_dqn_collector(self, dqn_policy, env):
        collector = SB3DQNDataCollector(SB3DQNDatapoint, dqn_policy)

        assert collector.datapoint_cls == SB3DQNDatapoint

        obs, _ = env.reset()
        datapoint, action = collector.collect_internal_data(obs)

        assert isinstance(datapoint, SB3DQNDatapoint)
        assert datapoint.observations is None
        assert datapoint.actions is None
        assert datapoint.rewards is None
        assert datapoint.terminateds is None
        assert datapoint.truncateds is None
        assert datapoint.steps is None
        assert datapoint.renders is None
        assert datapoint.q_vals is not None
        assert datapoint.latent_qs is not None
        assert datapoint.features is not None
        assert env.action_space.contains(action)

        new_obs, _, _, _, _ = env.step(action)
        _, action = collector.collect_internal_data(new_obs)
        assert env.action_space.contains(action)

        new_obs, _, _, _, _ = env.step(action)
        _, action = collector.collect_internal_data(new_obs)
        assert env.action_space.contains(action)

        new_obs, _, _, _, _ = env.step(action)
        _, action = collector.collect_internal_data(new_obs)
        assert env.action_space.contains(action)
