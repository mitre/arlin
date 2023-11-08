import os

import gymnasium as gym
from stable_baselines3.ppo import PPO

from arlin.dataset.loaders.sb3_loaders import load_sb_model


# from arlin.dataset.loaders.sb3_loaders import load_hf_sb_model, load_sb_model


class TestSB3Loaders:
    # def test_load_hf_sb_model(self):
    #     model = load_hf_sb_model(
    #         repo_id="sb3/ppo-LunarLander-v2",
    #         filename="ppo-LunarLander-v2.zip",
    #         algo_str="ppo",
    #     )

    #     assert isinstance(model, PPO)

    #     with pytest.raises(ValueError):
    #         model = load_hf_sb_model(repo_id="test", filename="invalid", algo_str="ppo")

    def test_load_sb_model(self, tmpdir):
        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=500)
        model_path = os.path.join(tmpdir, "test_model")
        model.save(model_path)
        del model

        model = load_sb_model(model_path, "ppo")
        assert isinstance(model, PPO)
