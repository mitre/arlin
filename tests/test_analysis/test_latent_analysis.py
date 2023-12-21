import numpy as np
import pytest

from arlin.analysis.latent_analysis import LatentAnalyzer
from arlin.analysis.visualization import COLORS, GraphData


@pytest.fixture
def analyzer(random_embeddings, random_dataset):
    analyzer = LatentAnalyzer(random_dataset, random_embeddings)
    return analyzer


@pytest.fixture
def ppo_analyzer(ppo_embeddings, ppo_dataset):
    ppo_analyzer = LatentAnalyzer(ppo_dataset, ppo_embeddings)
    return ppo_analyzer


class TestLatentAnalyzer:
    def test_init(self, random_dataset, random_embeddings):
        analyzer = LatentAnalyzer(random_dataset, random_embeddings)

        assert analyzer.num_embeddings == len(random_embeddings)
        assert len(analyzer.x) == len(random_embeddings)
        assert len(analyzer.y) == len(random_embeddings)

    def test_embeddings_graph_data(self, analyzer, random_embeddings):
        embeddings_data = analyzer.embeddings_graph_data()
        assert isinstance(embeddings_data, GraphData)

        assert len(embeddings_data.colors) == len(random_embeddings)
        assert embeddings_data.title == "Embeddings"
        assert embeddings_data.legend is None
        assert embeddings_data.cmap is None
        assert embeddings_data.xlabel is None
        assert embeddings_data.ylabel is None
        assert not embeddings_data.showall

    def test_clusters_graph_data(self, analyzer, random_clusters, random_embeddings):
        cluster_data = analyzer.clusters_graph_data(random_clusters[0])
        assert isinstance(cluster_data, GraphData)

        assert len(cluster_data.colors) == len(random_embeddings)
        for i, cluster_id in enumerate(random_clusters[0]):
            assert cluster_data.colors[i] == COLORS[cluster_id]

        n_clusters = len(np.unique(random_clusters[0]))
        assert cluster_data.title == f"{n_clusters} Clusters"

        assert cluster_data.legend["title"] == "Cluster Groups"
        assert cluster_data.legend["labels"] == [
            f"Cluster {i}" for i in range(n_clusters)
        ]
        assert len(cluster_data.legend["handles"]) == len(cluster_data.legend["labels"])
        assert cluster_data.cmap is None
        assert cluster_data.xlabel is None
        assert cluster_data.ylabel is None
        assert not cluster_data.showall

    def test_decision_boundary_graph_data(self, analyzer, random_embeddings):
        db_data = analyzer.decision_boundary_graph_data()
        assert isinstance(db_data, GraphData)

        assert len(db_data.colors) == len(random_embeddings)
        for i, action_id in enumerate(analyzer.dataset.actions):
            assert db_data.colors[i] == COLORS[action_id]

        assert db_data.title == "Decision Boundaries for Taken Actions"

        assert db_data.legend["title"] == "Action Values"
        assert db_data.legend["labels"] == ["0", "1", "2", "3"]
        assert len(db_data.legend["handles"]) == len(db_data.legend["labels"])
        assert db_data.cmap is None
        assert db_data.xlabel is None
        assert db_data.ylabel is None
        assert not db_data.showall

    def test_episode_prog_graph_data(self, analyzer, random_embeddings):
        prog_data = analyzer.episode_prog_graph_data()
        assert isinstance(prog_data, GraphData)

        assert len(prog_data.colors) == len(random_embeddings)
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

    def test_initial_terminal_state_data(self, analyzer, random_embeddings):
        it_data = analyzer.initial_terminal_state_data()
        assert isinstance(it_data, GraphData)

        assert len(it_data.colors) == len(random_embeddings)
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
        assert len(it_data.legend["handles"]) == len(it_data.legend["labels"])
        assert it_data.cmap is None
        assert it_data.xlabel is None
        assert it_data.ylabel is None
        assert not it_data.showall
