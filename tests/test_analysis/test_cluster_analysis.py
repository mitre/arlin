import os

import numpy as np
import pytest

from arlin.analysis.cluster_analysis import ClusterAnalyzer
from arlin.analysis.visualization import GraphData


@pytest.fixture
def analyzer(random_clusters, random_dataset):
    analyzer = ClusterAnalyzer(random_dataset, random_clusters[0])
    return analyzer


@pytest.fixture
def ppo_analyzer(ppo_clusters, ppo_dataset):
    analyzer = ClusterAnalyzer(ppo_dataset, ppo_clusters[0])
    return analyzer


class TestClusterAnalyzer:
    def test_init(self, analyzer, random_dataset, random_clusters):
        assert analyzer.num_clusters == max(analyzer.clusters) + 1
        assert len(analyzer.cluster_stage_colors) == analyzer.num_clusters

        for i, cluster_id in enumerate(analyzer.clusters):
            cluster_color = analyzer.cluster_stage_colors[cluster_id]
            if i in analyzer.dataset.start_indices:
                assert cluster_color == "g"
            elif i in analyzer.dataset.term_indices:
                assert cluster_color == "r"
            else:
                assert cluster_color == "k"

        random_clusters[0][random_dataset.start_indices[0]] = 0
        random_clusters[0][random_dataset.term_indices[0]] = 0
        with pytest.raises(ValueError):
            _ = ClusterAnalyzer(random_dataset, random_clusters[0])

    def test_cluster_state_img_analysis(self, analyzer, env, tmpdir):
        _, _ = env.reset()
        render = env.render()
        test_imgs = np.array([render] * 25)
        path = os.path.join(tmpdir, "cluster1")
        analyzer._cluster_state_img_analysis(test_imgs, 10, path)

        save_dir = os.path.join(path, "images")
        assert os.path.isdir(save_dir)
        assert len(os.listdir(save_dir)) == 10

        test_imgs = np.array([render] * 5)
        path = os.path.join(tmpdir, "cluster2")
        analyzer._cluster_state_img_analysis(test_imgs, 10, path)

        save_dir = os.path.join(path, "images")
        assert os.path.isdir(save_dir)
        assert len(os.listdir(save_dir)) == 5

    def test_cluster_state_text_analysis(self, analyzer, env, tmpdir):
        obs, _ = env.reset()
        cluster_indices = np.array([0, 1, 2, 3])
        cluster_states = [obs] * 4
        path = os.path.join(tmpdir, "test_dir")
        os.makedirs(path, exist_ok=True)

        analyzer._cluster_state_text_analysis(cluster_indices, cluster_states, env, path)

        assert os.path.isfile(os.path.join(path, "txt_analysis.txt"))

    def test_cluster_state_analysis(self, analyzer, env, tmpdir):
        path = os.path.join(tmpdir, "test_dir")
        analyzer.cluster_state_analysis(0, env, path, 10)

        cluster_0_size = len(np.where(analyzer.clusters == 0)[0])
        assert os.path.isdir(path)
        run_path = os.path.join(path, "cluster_0")
        assert len(os.listdir(os.path.join(run_path, "images"))) == cluster_0_size
        assert os.path.isfile(os.path.join(run_path, "txt_analysis.txt"))

    def test_cluster_confidence(self, analyzer, ppo_analyzer):
        with pytest.raises(ValueError):
            analyzer.cluster_confidence()

        cluster_conf_data = ppo_analyzer.cluster_confidence()

        assert isinstance(cluster_conf_data, GraphData)
        assert cluster_conf_data.colors == ppo_analyzer.cluster_stage_colors
        assert cluster_conf_data.title == "Cluster Confidence Analysis"
        assert cluster_conf_data.legend["title"] == "Cluster Stage"
        assert cluster_conf_data.legend["labels"] == [
            "Initial",
            "Intermediate",
            "Terminal",
        ]
        labels = cluster_conf_data.legend["labels"]
        assert len(cluster_conf_data.legend["handles"]) == len(labels)
        assert len(cluster_conf_data.x) == max(ppo_analyzer.clusters) + 1
        assert len(cluster_conf_data.y) == max(ppo_analyzer.clusters) + 1
        assert len(cluster_conf_data.error_bars) == max(ppo_analyzer.clusters) + 1
        assert cluster_conf_data.xlabel == "Cluster ID"
        assert cluster_conf_data.ylabel == "Mean Highest Action Confidence"
        assert cluster_conf_data.showall

    def test_cluster_rewards(self, analyzer):
        reward_data = analyzer.cluster_rewards()

        assert isinstance(reward_data, GraphData)
        assert reward_data.colors == analyzer.cluster_stage_colors
        assert reward_data.title == "Cluster Reward Analysis"
        assert reward_data.legend["title"] == "Cluster Stage"
        assert reward_data.legend["labels"] == ["Initial", "Intermediate", "Terminal"]
        labels = reward_data.legend["labels"]
        assert len(reward_data.legend["handles"]) == len(labels)
        assert len(reward_data.x) == max(analyzer.clusters) + 1
        assert len(reward_data.y) == max(analyzer.clusters) + 1
        assert len(reward_data.error_bars) == max(analyzer.clusters) + 1
        assert reward_data.xlabel == "Cluster ID"
        assert reward_data.ylabel == "Mean Reward"
        assert reward_data.showall

    def test_cluster_values(self, analyzer, ppo_analyzer):
        with pytest.raises(ValueError):
            _ = analyzer.cluster_values()

        value_data = ppo_analyzer.cluster_values()

        assert isinstance(value_data, GraphData)
        assert value_data.colors == ppo_analyzer.cluster_stage_colors
        assert value_data.title == "Cluster Value Analysis"
        assert value_data.legend["title"] == "Cluster Stage"
        assert value_data.legend["labels"] == ["Initial", "Intermediate", "Terminal"]
        labels = value_data.legend["labels"]
        assert len(value_data.legend["handles"]) == len(labels)
        assert len(value_data.x) == max(ppo_analyzer.clusters) + 1
        assert len(value_data.y) == max(ppo_analyzer.clusters) + 1
        assert len(value_data.error_bars) == max(ppo_analyzer.clusters) + 1
        assert value_data.xlabel == "Cluster ID"
        assert value_data.ylabel == "Mean Value"
        assert value_data.showall
