import pytest

from arlin.adversarial import attacks
from arlin.dataset.collectors import SB3PPODataCollector
from arlin.dataset.collectors.datapoints import SB3PPODatapoint


class TestAttacks:
    def test_should_attack(self):
        for x in ["Random", "Adversarial"]:
            assert not attacks.should_attack(
                x, timestep=0, freq=0, preference=0, threshold=1
            )

            assert not attacks.should_attack(
                x, timestep=10, freq=0, preference=0, threshold=1
            )

            assert attacks.should_attack(
                x, timestep=10, freq=1, preference=0, threshold=1
            )

            assert attacks.should_attack(
                x, timestep=10, freq=2, preference=0, threshold=1
            )

            assert not attacks.should_attack(
                x, timestep=10, freq=3, preference=0, threshold=1
            )

            assert not attacks.should_attack(
                x, timestep=10, freq=3, preference=0.6, threshold=0.5
            )

        assert not attacks.should_attack(
            "Preference", timestep=10, freq=2, preference=0.4, threshold=0.5
        )

        assert not attacks.should_attack(
            "Preference", timestep=10, freq=2, preference=0.5, threshold=0.5
        )

        assert attacks.should_attack(
            "Preference", timestep=10, freq=2, preference=0.6, threshold=0.5
        )

        with pytest.raises(ValueError):
            attacks.should_attack("test", timestep=1)

    def test_random_action(self, env):
        action_space = env.action_space.n
        assert attacks.random_action(env) < action_space

    def test_adversarial_action(self, env, ppo_model):
        action_space = env.action_space.n
        obs, _ = env.reset()
        assert attacks.adversarial_action(obs, ppo_model) < action_space

    def test_run_baseline(self, env, ppo_model):
        obs, rewards, renders = attacks.run_baseline(env, ppo_model, 2)

        assert len(obs) == 2
        assert len(rewards) == 2
        assert len(renders) == 2

    def test_run_adversarial(self, env, ppo_model):
        collector = SB3PPODataCollector(
            datapoint_cls=SB3PPODatapoint, policy=ppo_model.policy
        )

        obs, rewards, kl_divs, renders = attacks.run_adversarial(
            "random", collector, env, ppo_model, ppo_model, attack_freq=2, num_episodes=2
        )

        assert len(obs) == 2
        assert len(rewards) == 2
        assert len(kl_divs) > 0
        assert len(renders) == 2

    # TODO: How to make run_arlin modular and not hardcode clustering as latents
    # def test_run_arlin(self, env, ppo_model, ppo_clusters):
    #     collector = SB3PPODataCollector(datapoint_cls=SB3PPODatapoint,
    #                                     policy=ppo_model.policy)

    #     obs, rewards, kl_divs, renders = attacks.run_arlin(collector,
    #                                                  env,
    #                                                  ppo_model,
    #                                                  ppo_clusters[1],
    #                                                  ppo_clusters[2],
    #                                                  ppo_clusters[3],
    #                                                  num_episodes=2)

    #     assert len(obs) == 2
    #     assert len(rewards) == 2
    #     assert len(kl_divs) > 0
    #     assert len(renders) == 2
