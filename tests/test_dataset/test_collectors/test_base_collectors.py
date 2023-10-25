from arlin.dataset.collectors import RandomDataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint


class TestRandomCollectors:
    def test_random_collector(self, env):
        collector = RandomDataCollector(BaseDatapoint, env)

        assert collector.datapoint_cls == BaseDatapoint

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
