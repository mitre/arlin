import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from arlin.dataset import XRLDataset
from arlin.dataset.collectors import RandomDataCollector, SB3PPODataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint, SB3PPODatapoint
from arlin.generation import generate_clusters, generate_embeddings


@pytest.fixture
def env():
    # Create environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    return env


@pytest.fixture
def random_dataset(env):
    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)
    dataset.fill(num_datapoints=50, randomness=0.25)

    return dataset


@pytest.fixture
def random_embeddings(random_dataset):
    embeddings = generate_embeddings(
        dataset=random_dataset,
        activation_key="observations",
        perplexity=5,
        n_train_iter=250,
        output_dim=2,
        seed=12345,
    )

    return embeddings


@pytest.fixture
def random_clusters(random_dataset):
    clusters, _, _, _ = generate_clusters(
        random_dataset,
        ["observations", "rewards"],
        ["observations", "rewards"],
        ["rewards"],
        10,
        seed=1234,
    )
    return clusters


@pytest.fixture
def ppo_dataset(env):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(100))

    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = SB3PPODataCollector(datapoint_cls=SB3PPODatapoint, policy=model.policy)

    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)
    dataset.fill(num_datapoints=50, randomness=0.25)

    return dataset


@pytest.fixture
def ppo_embeddings(ppo_dataset):
    embeddings = generate_embeddings(
        dataset=ppo_dataset,
        activation_key="latent_actors",
        perplexity=5,
        n_train_iter=250,
        output_dim=2,
        seed=12345,
    )

    return embeddings


@pytest.fixture
def ppo_clusters(ppo_dataset):
    clusters, _, _, _ = generate_clusters(
        ppo_dataset,
        ["observations", "rewards"],
        ["observations", "rewards"],
        ["rewards"],
        10,
        seed=1234,
    )
    return clusters
