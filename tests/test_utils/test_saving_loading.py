import os

import gymnasium as gym
import numpy as np
import pytest

import arlin.dataset.loaders as loaders
from arlin.dataset import XRLDataset
from arlin.dataset.collectors import SB3PPODataCollector
from arlin.dataset.collectors.datapoints import SB3PPODatapoint
from arlin.generation import generate_embeddings
from arlin.utils.saving_loading import load_data, save_data


@pytest.fixture
def dataset():
    # Create environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    # Load the SB3 model from Huggingface
    model = loaders.load_hf_sb_model(
        repo_id="sb3/ppo-LunarLander-v2",
        filename="ppo-LunarLander-v2.zip",
        algo_str="ppo",
    )

    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = SB3PPODataCollector(datapoint_cls=SB3PPODatapoint, policy=model.policy)

    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)
    dataset.fill(num_datapoints=50, randomness=0.25)

    return dataset


@pytest.fixture
def embeddings(dataset):
    embeddings = generate_embeddings(
        dataset=dataset,
        activation_key="latent_actors",
        perplexity=5,
        n_train_iter=250,
        output_dim=2,
        seed=12345,
    )

    return embeddings


class TestSavingLoading:
    def test_save_data(self, tmpdir, embeddings):
        embeddings_path = os.path.join(tmpdir, "embeddings.npy")
        embeddings_2_path = os.path.join(tmpdir, "tests", "embeddings_2.npy")
        save_data(embeddings, os.path.join(tmpdir, "embeddings"))
        save_data(embeddings, os.path.join(tmpdir, "tests", "embeddings_2.npy"))

        assert os.path.isfile(embeddings_path)
        assert os.path.isfile(embeddings_2_path)

    def test_load_data(self, tmpdir, embeddings):
        embeddings_path = os.path.join(tmpdir, "embeddings.npy")
        save_data(embeddings, embeddings_path)
        loaded_embeddings = load_data(embeddings_path)

        assert np.array_equal(embeddings, loaded_embeddings)
