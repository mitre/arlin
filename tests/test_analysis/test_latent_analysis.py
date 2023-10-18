import gymnasium as gym
import numpy as np
import pytest

from arlin.analysis.latent_analysis import LatentAnalyzer
from arlin.analysis.visualization import COLORS, GraphData
from arlin.dataset import XRLDataset
from arlin.dataset.collectors import RandomDataCollector, SB3PPODataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint, SB3PPODatapoint
from arlin.dataset.loaders import load_hf_sb_model
from arlin.generation import generate_clusters, generate_embeddings


@pytest.fixture
def dataset():
    # Create environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)
    dataset.fill(num_datapoints=50, randomness=0.25)

    return dataset


@pytest.fixture
def embeddings(dataset):
    embeddings = generate_embeddings(dataset, "observations", 10, 20)
    return embeddings


@pytest.fixture
def analyzer(embeddings, dataset):
    analyzer = LatentAnalyzer(embeddings, dataset)
    return analyzer


@pytest.fixture
def clusters(dataset):
    clusters, _, _, _ = generate_clusters(
        dataset,
        ["observations", "rewards"],
        ["observations", "rewards"],
        ["rewards"],
        10,
        seed=1234,
    )
    return clusters


@pytest.fixture
def ppo_dataset():
    # Create environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    # Load the SB3 model from Huggingface
    model = load_hf_sb_model(
        repo_id="sb3/ppo-LunarLander-v2",
        filename="ppo-LunarLander-v2.zip",
        algo_str="ppo",
    )

    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = SB3PPODataCollector(datapoint_cls=SB3PPODatapoint, policy=model.policy)

    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)
    dataset.fill(num_datapoints=50, randomness=0.0)
    return dataset


@pytest.fixture
def ppo_embeddings(ppo_dataset):
    embeddings = generate_embeddings(ppo_dataset, "observations", 10, 20)
    return embeddings


@pytest.fixture
def ppo_analyzer(ppo_embeddings, ppo_dataset):
    ppo_analyzer = LatentAnalyzer(ppo_embeddings, ppo_dataset)
    return ppo_analyzer


class TestLatentAnalyzer:
    def test_init(self, dataset, embeddings):
        analyzer = LatentAnalyzer(embeddings, dataset)

        assert analyzer.num_embeddings == len(embeddings)
        assert len(analyzer.x) == len(embeddings)
        assert len(analyzer.y) == len(embeddings)

    def test_embeddings_graph_data(self, analyzer, embeddings):
        embeddings_data = analyzer.embeddings_graph_data()
        assert isinstance(embeddings_data, GraphData)

        assert len(embeddings_data.colors) == len(embeddings)
        assert embeddings_data.title == "Embeddings"
        assert embeddings_data.legend is None
        assert embeddings_data.cmap is None
        assert embeddings_data.xlabel is None
        assert embeddings_data.ylabel is None
        assert not embeddings_data.showall

    def test_clusters_graph_data(self, analyzer, clusters, embeddings):
        cluster_data = analyzer.clusters_graph_data(clusters)
        assert isinstance(cluster_data, GraphData)

        assert len(cluster_data.colors) == len(embeddings)
        for i, cluster_id in enumerate(clusters):
            assert cluster_data.colors[i] == COLORS[cluster_id]

        n_clusters = len(np.unique(clusters))
        assert cluster_data.title == f"{n_clusters} Clusters"

        assert cluster_data.legend["title"] == "Cluster Groups"
        assert cluster_data.legend["labels"] == [
            f"Cluster {i}" for i in range(n_clusters)
        ]
        assert cluster_data.cmap is None
        assert cluster_data.xlabel is None
        assert cluster_data.ylabel is None
        assert not cluster_data.showall

    def test_decision_boundary_graph_data(self, analyzer, embeddings):
        db_data = analyzer.decision_boundary_graph_data()
        assert isinstance(db_data, GraphData)

        assert len(db_data.colors) == len(embeddings)
        for i, action_id in enumerate(analyzer.dataset.actions):
            assert db_data.colors[i] == COLORS[action_id]

        assert db_data.title == "Decision Boundaries for Taken Actions"

        assert db_data.legend["title"] == "Action Values"
        assert db_data.legend["labels"] == ["0", "1", "2", "3"]
        assert db_data.cmap is None
        assert db_data.xlabel is None
        assert db_data.ylabel is None
        assert not db_data.showall

    def test_episode_prog_graph_data(self, analyzer, embeddings):
        prog_data = analyzer.episode_prog_graph_data()
        assert isinstance(prog_data, GraphData)

        assert len(prog_data.colors) == len(embeddings)
        assert prog_data.title == "Episode Progression"

        assert prog_data.legend is None
        assert prog_data.cmap == "viridis"
        assert prog_data.xlabel is None
        assert prog_data.ylabel is None
        assert not prog_data.showall

    def test_confidence_data(self, analyzer, ppo_embeddings, ppo_analyzer):
        with pytest.raises(ValueError):
            _ = analyzer.confidence_data()

        conf_data = ppo_analyzer.confidence_data()
        assert isinstance(conf_data, GraphData)
        assert len(conf_data.colors) == len(ppo_embeddings)
        assert conf_data.title == "Policy Confidence in Greedy Action"

        assert conf_data.legend is None
        assert conf_data.cmap == "RdYlGn"
        assert conf_data.xlabel is None
        assert conf_data.ylabel is None
        assert not conf_data.showall

    def test_initial_terminal_state_data(self, analyzer, embeddings):
        it_data = analyzer.initial_terminal_state_data()
        assert isinstance(it_data, GraphData)

        assert len(it_data.colors) == len(embeddings)
        for i, val in enumerate(analyzer.dataset.terminateds):
            if val == 1:
                assert it_data.colors[i] == COLORS[0]
            elif i in analyzer.dataset.start_indices:
                assert it_data.colors[i] == COLORS[1]
            else:
                assert it_data.colors[i] == "#F5F5F5"

        assert it_data.title == "Initial and Terminal States"

        assert it_data.legend["title"] == "State Type"
        assert it_data.legend["labels"] == ["Initial", "Terminal"]
        assert it_data.cmap is None
        assert it_data.xlabel is None
        assert it_data.ylabel is None
        assert not it_data.showall
