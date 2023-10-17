import gymnasium as gym
import numpy as np
import pytest

from arlin.dataset import XRLDataset
from arlin.dataset.collectors import RandomDataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint
from arlin.generation import (
    _get_cluster_ons,
    _select_key_data,
    generate_clusters,
    generate_embeddings,
)


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


class TestGeneration:
    def test_generate_embeddings(self, dataset):
        embeddings = generate_embeddings(dataset, "observations", 10, 20)

        assert len(embeddings) == len(dataset.observations)

    def test_select_key_data(self, dataset):
        start_data = _select_key_data(
            dataset, ["observations", "rewards"], dataset.start_indices
        )

        assert len(start_data) == 2
        assert len(start_data[0]) == len(start_data[1]) == len(dataset.start_indices)

    def test_get_cluster_ons(self, dataset):
        cluster_on_start, cluster_on_mid, cluster_on_term, mid_mask = _get_cluster_ons(
            dataset, ["observations", "rewards"], ["observations", "rewards"], ["rewards"]
        )

        assert len(cluster_on_start) == len(dataset.start_indices)
        assert len(cluster_on_term) == len(dataset.term_indices)
        num_cluster_on = (
            len(dataset.observations)
            - len(dataset.term_indices)
            - len(dataset.start_indices)
        )
        assert len(cluster_on_mid) == num_cluster_on
        assert sum(mid_mask) == len(cluster_on_mid)

    def test_generate_clusters(self, dataset):
        clusters, start_algo, mid_algo, term_algo = generate_clusters(
            dataset,
            ["observations", "rewards"],
            ["observations", "rewards"],
            ["rewards"],
            10,
            seed=1234,
        )

        num_start_clusters = len(start_algo.cluster_centers_)
        num_mid_clusters = len(mid_algo.cluster_centers_)
        num_term_clusters = len(term_algo.cluster_centers_)

        assert num_mid_clusters == 10
        total_clusters = num_mid_clusters + num_term_clusters + num_start_clusters
        num_clusters = len(np.unique(clusters))
        assert total_clusters == num_clusters

        for i, cluster_id in enumerate(clusters):
            assert 0 <= cluster_id < num_clusters

            if i in dataset.start_indices:
                assert 10 <= cluster_id < 10 + num_start_clusters
            elif i in dataset.term_indices:
                assert 10 + num_start_clusters <= cluster_id < total_clusters
            else:
                assert 0 <= cluster_id < 10
