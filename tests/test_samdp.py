import os

import gymnasium as gym
import networkx as nx
import numpy as np
import pytest

from arlin.analysis.visualization import COLORS
from arlin.dataset.collectors import RandomDataCollector
from arlin.dataset.collectors.datapoints import BaseDatapoint
from arlin.dataset.xrl_dataset import XRLDataset
from arlin.generation import generate_clusters
from arlin.samdp import SAMDP


@pytest.fixture
def env():
    # Create environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    return env


@pytest.fixture
def dataset(env):
    # Create the datapoint collector for SB3 PPO Datapoints with the model's policy
    collector = RandomDataCollector(datapoint_cls=BaseDatapoint, environment=env)
    # Instantiate the XRL Dataset
    dataset = XRLDataset(env, collector=collector)
    dataset.fill(num_datapoints=50, randomness=0.25)

    return dataset


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
def samdp(dataset, clusters):
    samdp = SAMDP(clusters, dataset)
    return samdp


class TestSAMDP:
    def test_init(self, clusters, dataset):
        samdp = SAMDP(clusters, dataset)
        assert np.array_equal(samdp.clusters, clusters)
        assert samdp.dataset == dataset

    def test_generate(self, samdp):
        samdp_obj = samdp._generate()

        num_clusters = max(samdp.clusters) + 1
        num_actions = samdp.dataset.env.action_space.n

        assert samdp_obj.shape == (num_clusters, num_actions + 1, num_clusters)

        for cluster_id in range(num_clusters):
            for action in range(num_actions):
                if sum(samdp_obj[cluster_id][action]) != 0:
                    assert sum(samdp_obj[cluster_id][action]) == 1

        assert np.sum(np.isnan(samdp_obj)) == 0

        for i in range(samdp.dataset.num_datapoints):
            if samdp.dataset.terminateds[i] or samdp.dataset.truncateds[i]:
                continue
            action = samdp.dataset.actions[i]
            from_cluster = samdp.clusters[i]
            to_cluster = samdp.clusters[i + 1]

            if from_cluster != to_cluster:
                action_prob = samdp.samdp_counts[from_cluster][action][to_cluster] / sum(
                    samdp.samdp_counts[from_cluster][action][:]
                )
                assert samdp_obj[from_cluster][action][to_cluster] == action_prob
                total_prob = samdp.samdp_counts[from_cluster][-1][to_cluster] / sum(
                    samdp.samdp_counts[from_cluster][-1][:]
                )
                assert samdp_obj[from_cluster][-1][to_cluster] == total_prob

    def test_save_txt(self, samdp, tmpdir):
        path = os.path.join(tmpdir)
        samdp.save_txt(path)

        assert os.path.isfile(os.path.join(path, "samdp.txt"))

    def test_generate_graph(self, samdp):
        graph = samdp._generate_graph()

        num_clusters = max(samdp.clusters) + 1
        assert graph.number_of_nodes() == num_clusters

        start_clusters = set(samdp.clusters[samdp.dataset.start_indices])
        term_clusters = set(samdp.clusters[samdp.dataset.term_indices])

        nodes = graph.nodes(data=True)
        for i in range(num_clusters):
            if i in start_clusters:
                edge_color = "g"
            elif i in term_clusters:
                edge_color = "r"
            else:
                edge_color = "k"

            node = (f"Cluster {i}", {"edge_color": edge_color, "color": COLORS[i]})

            assert node in nodes

        edges = graph.edges(data=True)
        for i in range(samdp.dataset.num_datapoints):
            if samdp.dataset.terminateds[i] or samdp.dataset.truncateds[i]:
                continue
            action = samdp.dataset.actions[i]
            from_cluster = samdp.clusters[i]
            to_cluster = samdp.clusters[i + 1]

            if from_cluster == to_cluster:
                continue

            prob = samdp.samdp[from_cluster, action, to_cluster]
            edge = (
                f"Cluster {from_cluster}",
                f"Cluster {to_cluster}",
                {"weight": prob, "action": action, "color": COLORS[action]},
            )

            assert edge in edges

    def test_generate_simplified_graph(self, samdp):
        graph = samdp._generate_simplified_graph()

        num_clusters = max(samdp.clusters) + 1
        assert graph.number_of_nodes() == num_clusters

        start_clusters = set(samdp.clusters[samdp.dataset.start_indices])
        term_clusters = set(samdp.clusters[samdp.dataset.term_indices])

        nodes = graph.nodes(data=True)
        for i in range(num_clusters):
            if i in start_clusters:
                edge_color = "g"
            elif i in term_clusters:
                edge_color = "r"
            else:
                edge_color = "k"

            node = (f"Cluster {i}", {"edge_color": edge_color, "color": COLORS[i]})

            assert node in nodes

        edges = graph.edges(data=True)
        for i in range(samdp.dataset.num_datapoints):
            if samdp.dataset.terminateds[i] or samdp.dataset.truncateds[i]:
                continue
            from_cluster = samdp.clusters[i]
            to_cluster = samdp.clusters[i + 1]

            if from_cluster == to_cluster:
                continue

            prob = np.sum(samdp.samdp[from_cluster, -1, to_cluster])
            edge = (
                f"Cluster {from_cluster}",
                f"Cluster {to_cluster}",
                {"weight": prob, "action": -1, "color": "#000000"},
            )

            assert edge in edges

    def test_set_node_attributes(self):
        pass  # test in test_generate_graph()

    def test_generate_bfs_pos(self, samdp):
        pos = samdp._generate_bfs_pos()

        num_clusters = max(samdp.clusters) + 1
        start_clusters = set(samdp.clusters[samdp.dataset.start_indices])
        term_clusters = set(samdp.clusters[samdp.dataset.term_indices])
        initial_nodes = [f"Cluster {i}" for i in start_clusters]
        terminal_nodes = [f"Cluster {i}" for i in term_clusters]

        init_y = set()
        for node in initial_nodes:
            assert pos[node][0] == 0
            init_y.add(pos[node][1])

        assert len(init_y) == len(initial_nodes)

        for i, y in enumerate(sorted(init_y)):
            assert i == y

        depth = set()
        for node in samdp.graph.nodes():
            assert pos[node]
            depth.add(pos[node][0])

        locs = set()
        for node in samdp.graph.nodes():
            locs.add(pos[node])
            if node not in initial_nodes and node not in terminal_nodes:
                assert pos[node][0] not in [0, max(depth)]

        assert len(locs) == num_clusters

        term_y = set()
        for node in terminal_nodes:
            assert pos[node][0] == max(depth)
            term_y.add(pos[node][1])

        assert len(term_y) == len(terminal_nodes)

        for i, y in enumerate(sorted(term_y)):
            assert i == y

    def test_generate_edge_arcs(self, samdp):
        pos = samdp._generate_bfs_pos()
        edges = samdp.graph.edges(keys=True)
        edge_arcs = samdp._generate_edge_arcs(pos, edges)

        for i, edge in enumerate(edges):
            loc_1_x = pos[edge[0]][0]
            loc_1_y = pos[edge[0]][1]
            loc_2_x = pos[edge[1]][0]
            loc_2_y = pos[edge[1]][1]
            arc = edge_arcs[i]
            edge_key = edge[2]

            reverse_edge = (edge[1], edge[0]) in edges

            if loc_1_x == loc_2_x:
                assert arc == (0.05 * (edge_key + 1))
            elif loc_1_y == loc_2_y:
                assert arc == (0.05 * (edge_key + 1))
            elif reverse_edge:
                assert arc == (0.05 * (edge_key + 1))
            else:
                assert arc == 0

    def test_save_complete_graph(self, samdp, tmpdir):
        path = os.path.join(tmpdir, "complete_graph.png")
        graph = samdp.save_complete_graph(path)

        assert graph == samdp.graph
        assert os.path.isfile(path)

    def test_save_simplified_graph(self, samdp, tmpdir):
        path = os.path.join(tmpdir, "simplified_graph.png")
        _ = samdp.save_simplified_graph(path)

        assert os.path.isfile(path)

    def test_save_likely_graph(self, samdp, tmpdir):
        path = os.path.join(tmpdir, "likely_graph.png")
        graph = samdp.save_likely_graph(path)
        assert os.path.isfile(path)

        num_clusters = max(samdp.clusters) + 1
        assert graph.number_of_nodes() == num_clusters

        start_clusters = set(samdp.clusters[samdp.dataset.start_indices])
        term_clusters = set(samdp.clusters[samdp.dataset.term_indices])

        nodes = graph.nodes(data=True)
        for i in range(num_clusters):
            if i in start_clusters:
                edge_color = "g"
            elif i in term_clusters:
                edge_color = "r"
            else:
                edge_color = "k"

            node = (f"Cluster {i}", {"edge_color": edge_color, "color": COLORS[i]})

            assert node in nodes

        edges = graph.edges(data=True, keys=True)

        for edge in edges:
            from_cluster_id = int(edge[0].split(" ")[-1])
            action = edge[3]["action"]

            to_cluster = np.argmax(samdp.samdp[from_cluster_id, action, :])

            assert edge[1] == f"Cluster {to_cluster}"
            assert edge[3]["weight"] == samdp.samdp[from_cluster_id, action, to_cluster]

    def test_find_best_path(self, samdp):
        from_cluster = "Cluster 9"
        to_cluster = "Cluster 11"

        best_path = samdp._find_best_path(9, 11, [])
        assert best_path == []

        paths = list(nx.all_simple_edge_paths(samdp.graph, from_cluster, to_cluster))

        updated_paths = []
        for path in paths:
            data_path = []
            for edge in path:
                edge_data = samdp.graph.get_edge_data(edge[0], edge[1], edge[2])
                updated_edge = (edge[0], edge[1], edge[2], edge_data)
                data_path.append(updated_edge)
            updated_paths.append(data_path)

        best_path = samdp._find_best_path(9, 11, updated_paths)

        path_probs = []
        for path in updated_paths:
            prob = 1
            for i, edge in enumerate(path):
                from_cluster_id = int(edge[0].split(" ")[-1])
                to_cluster_id = int(edge[1].split(" ")[-1])
                action = edge[3]["action"]
                if i == 0:
                    prob *= samdp.samdp[from_cluster_id, action, to_cluster_id]
                else:
                    prob *= np.sum(samdp.samdp[from_cluster_id, -1, to_cluster_id])
            path_probs.append(prob)

        best_prob = max(path_probs)

        out_prob = 1
        for i, edge in enumerate(best_path):
            if i == 0:
                out_prob *= edge[3]["weight"]
            else:
                from_cluster = int(edge[0].split(" ")[-1])
                to_cluster = int(edge[1].split(" ")[-1])
                out_prob *= np.sum(samdp.samdp[from_cluster, -1, to_cluster])

        assert best_prob == out_prob

    def test_save_paths(self, samdp, tmpdir):
        save_path = os.path.join(tmpdir, "paths.png")
        samdp.save_paths(-1, 0, save_path)
        assert not os.path.isfile(save_path)

        samdp.save_paths(0, -1, save_path)
        assert not os.path.isfile(save_path)

        samdp.save_paths(0, 0, save_path)
        assert not os.path.isfile(save_path)

        samdp.save_paths(0, 10, save_path)
        assert not os.path.isfile(save_path)

        samdp.save_paths(0, 10, save_path)
        assert not os.path.isfile(save_path)

        samdp.save_paths(9, 11, save_path)
        assert os.path.isfile(save_path)

        save_path = os.path.join(tmpdir, "paths_bp.png")
        samdp.save_paths(9, 11, save_path, best_path_only=True)
        assert os.path.isfile(save_path)

        save_path = os.path.join(tmpdir, "paths_verbose.png")
        samdp.save_paths(9, 11, save_path, verbose=True)
        assert os.path.isfile(save_path)

        save_path = os.path.join(tmpdir, "paths_verbose_bp.png")
        samdp.save_paths(9, 11, save_path, best_path_only=True, verbose=True)
        assert os.path.isfile(save_path)

    def test_save_terminal_paths(self, samdp, tmpdir):
        save_path = os.path.join(tmpdir, "paths.png")
        samdp.save_terminal_paths(save_path, term_cluster_id=-1)
        assert not os.path.isfile(save_path)

        samdp.save_terminal_paths(save_path, term_cluster_id=0)
        assert not os.path.isfile(save_path)

        samdp.save_terminal_paths(save_path, term_cluster_id=11)
        assert os.path.isfile(save_path)

        save_path = os.path.join(tmpdir, "path_11_bp.png")
        samdp.save_terminal_paths(save_path, best_path=True, term_cluster_id=11)
        assert os.path.isfile(save_path)

        save_path = os.path.join(tmpdir, "paths_bp.png")
        samdp.save_terminal_paths(save_path, best_path=True)
        assert os.path.isfile(save_path)

        save_path = os.path.join(tmpdir, "paths_11_bp.png")
        samdp.save_terminal_paths(save_path, best_path=True, term_cluster_id=11)
        assert os.path.isfile(save_path)

    def test_save_all_paths_to(self, tmpdir, samdp):
        path = os.path.join(tmpdir, "path_to.png")
        samdp.save_all_paths_to(-1, path)
        assert not os.path.isfile(path)

        samdp.save_all_paths_to(11, path)
        assert os.path.isfile(path)

        path = os.path.join(tmpdir, "path_to_verbose.png")
        samdp.save_all_paths_to(11, path, verbose=True)
        assert os.path.isfile(path)
