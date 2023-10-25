import pytest
from stable_baselines3 import DQN

from arlin.dataset.collectors import SB3DQNDataCollector, SB3PPODataCollector
from arlin.dataset.collectors.datapoints import SB3DQNDatapoint, SB3PPODatapoint


@pytest.fixture
def dqn_model(env):
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(100))
    return model


class TestSB3Collectors:
    def test_sb3_ppo_collector(self, ppo_model, env):
        collector = SB3PPODataCollector(SB3PPODatapoint, ppo_model.policy)

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

    def test_sb3_dqn_collector(self, dqn_model, env):
        collector = SB3DQNDataCollector(SB3DQNDatapoint, dqn_model.policy)

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
