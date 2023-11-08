import numpy as np

from arlin.generation import (
    _get_cluster_ons,
    _select_key_data,
    generate_clusters,
    generate_embeddings,
)


class TestGeneration:
    def test_generate_embeddings(self, random_dataset):
        embeddings = generate_embeddings(random_dataset, "observations", 10, 250)

        assert len(embeddings) == len(random_dataset.observations)

    def test_select_key_data(self, random_dataset):
        start_data = _select_key_data(
            random_dataset, ["observations", "rewards"], random_dataset.start_indices
        )

        assert len(start_data) == 2
        assert (
            len(start_data[0]) == len(start_data[1]) == len(random_dataset.start_indices)
        )

    def test_get_cluster_ons(self, random_dataset):
        cluster_on_start, cluster_on_mid, cluster_on_term, mid_mask = _get_cluster_ons(
            random_dataset,
            ["observations", "rewards"],
            ["observations", "rewards"],
            ["rewards"],
        )

        assert len(cluster_on_start) == len(random_dataset.start_indices)
        assert len(cluster_on_term) == len(random_dataset.term_indices)
        num_cluster_on = (
            len(random_dataset.observations)
            - len(random_dataset.term_indices)
            - len(random_dataset.start_indices)
        )
        assert len(cluster_on_mid) == num_cluster_on
        assert sum(mid_mask) == len(cluster_on_mid)

    def test_generate_clusters(self, random_dataset):
        clusters, start_algo, mid_algo, term_algo = generate_clusters(
            random_dataset,
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

            if i in random_dataset.start_indices:
                assert 10 <= cluster_id < 10 + num_start_clusters
            elif i in random_dataset.term_indices:
                assert 10 + num_start_clusters <= cluster_id < total_clusters
            else:
                assert 0 <= cluster_id < 10
