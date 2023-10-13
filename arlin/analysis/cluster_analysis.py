import logging
import os
import statistics
from typing import List

import gymnasium as gym
import numpy as np
from matplotlib.patches import Patch
from PIL import Image
from prettytable import PrettyTable
from tqdm import tqdm

from arlin.analysis.visualization import GraphData
from arlin.dataset import XRLDataset


class ClusterAnalyzer:
    def __init__(self, dataset: XRLDataset, clusters: np.ndarray):
        self.dataset = dataset
        self.clusters = clusters
        self.num_clusters = len(np.unique(self.clusters))

        start_clusters = set(self.clusters[self.dataset.start_indices])
        term_clusters = set(self.clusters[self.dataset.term_indices])

        self.cluster_stage_colors = []

        for cluster_id in range(self.num_clusters):
            if cluster_id in start_clusters and cluster_id in term_clusters:
                self.cluster_stage_colors.append("y")
            elif cluster_id in start_clusters:
                self.cluster_stage_colors.append("g")
            elif cluster_id in term_clusters:
                self.cluster_stage_colors.append("r")
            else:
                self.cluster_stage_colors.append("k")

    def cluster_state_img_analysis(
        self, cluster_renders: np.ndarray, num_samples: int, cluster_dir: str
    ):
        os.makedirs(os.path.join(cluster_dir, "images"), exist_ok=True)
        save_dir = os.path.join(cluster_dir, "images")

        sample_indices = np.random.randint(
            low=0, high=len(cluster_renders), size=num_samples
        )

        for i in tqdm(sample_indices):
            im = Image.fromarray(cluster_renders[i])
            im.save(os.path.join(save_dir, f"image_{i}.png"))

    def cluster_state_analysis(
        self,
        cluster_id: int,
        env: gym.Env,
        save_dir_path: str,
        num_img_samples: int = 10,
    ):
        cluster_run = os.path.join(save_dir_path, f"cluster_{cluster_id}")
        os.makedirs(cluster_run, exist_ok=True)

        cluster_indices = np.where(self.clusters == cluster_id)[0]
        cluster_states = self.dataset.observations[cluster_indices]
        cluster_renders = self.dataset.renders[cluster_indices]

        logging.info(f"State analysis of cluster {cluster_id} saved to {cluster_run}.")
        self.cluster_state_text_analysis(
            cluster_indices, cluster_states, env, cluster_run
        )
        logging.info(f"Saving {num_img_samples} images from Cluster {cluster_id}.")
        self.cluster_state_img_analysis(cluster_renders, num_img_samples, cluster_run)

    def cluster_state_text_analysis(
        self,
        cluster_indices: np.ndarray,
        cluster_states: List[np.ndarray],
        env: gym.Env,
        save_dir: str,
    ) -> None:
        obs_highs = env.observation_space.high
        obs_lows = env.observation_space.low
        obs_dim = env.observation_space.shape[0]

        cluster_size = len(cluster_states)

        data = np.empty([cluster_size, obs_dim])
        for e, state in enumerate(cluster_states):
            for i in range(obs_dim):
                data[e][i] = state[i]

        means = np.round(np.mean(data, axis=0), 4)
        stdevs = np.round(np.std(data, axis=0), 4)

        cluster_steps = self.dataset.steps[cluster_indices]

        step_mean = round(statistics.mean(cluster_steps), 2)
        try:
            step_stdev = round(statistics.stdev(cluster_steps), 2)
        except Exception:
            step_stdev = 0

        table = PrettyTable()
        table.title = "State Analysis by Cluster"

        table.field_names = ["Obs Dim", "Dim Range", "Mean", "Stdev"]

        for dim in range(obs_dim):
            mean = round(means[dim], 4)
            stdev = round(stdevs[dim], 4)
            row = [
                f"Obs Dim {dim}",
                f"[{obs_lows[dim]}, {obs_highs[dim]}]",
                f"{mean}",
                f"{stdev}",
            ]
            table.add_row(row)

        txt_data = [
            f"Cluster {os.path.basename(save_dir)}",
            f"Cluster Size: {len(cluster_states)}",
            f"Average Step: {step_mean}",
            f"Step Variance: {step_stdev}",
        ]

        txt_data.append(str(table))
        txt_data = "\n".join(txt_data)

        text_file_path = os.path.join(save_dir, "txt_analysis.txt")
        with open(text_file_path, "w") as f:
            f.write(txt_data)

    def cluster_confidence(self) -> GraphData:
        cluster_conf = [[] for _ in range(self.num_clusters)]

        for e, i in enumerate(self.clusters):
            conf = np.amax(self.dataset.dist_probs[e]).astype(np.float64)
            cluster_conf[i].append(conf)

        means = []
        stdevs = []

        for i in range(self.num_clusters):
            means.append(statistics.mean(cluster_conf[i]))

            try:
                stdevs.append(statistics.stdev(cluster_conf[i]))
            except Exception:
                stdevs.append(0)

        title = "Cluster Confidence Analysis"

        handles = [Patch(color="g"), Patch(color="r"), Patch(color="k")]
        labels = ["Initial", "Terminal", "Intermediate"]
        leg_title = "Cluster Stage"
        legend = {"handles": handles, "labels": labels, "title": leg_title}

        cluster_conf_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            legend=legend,
            error_bars=stdevs,
            xlabel="Cluster ID",
            ylabel="Mean Highest Action Confidence",
            showall=True,
        )

        return cluster_conf_data

    def cluster_rewards(self) -> GraphData:
        cluster_reward = [[] for _ in range(self.num_clusters)]

        for e, i in enumerate(self.clusters):
            total_rew = self.dataset.rewards[e].astype(np.float64)
            cluster_reward[i].append(total_rew)

        means = []
        stdevs = []

        for i in range(self.num_clusters):
            means.append(statistics.mean(cluster_reward[i]))

            try:
                stdevs.append(statistics.stdev(cluster_reward[i]))
            except Exception:
                stdevs.append(0)

        title = "Cluster Reward Analysis"

        handles = [Patch(color="g"), Patch(color="r"), Patch(color="k")]
        labels = ["Initial", "Terminal", "Intermediate"]
        leg_title = "Cluster Stage"
        legend = {"handles": handles, "labels": labels, "title": leg_title}

        cluster_reward_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            legend=legend,
            error_bars=stdevs,
            xlabel="Cluster ID",
            ylabel="Mean Reward",
            showall=True,
        )

        return cluster_reward_data

    def cluster_values(self) -> GraphData:
        cluster_value = [[] for _ in range(self.num_clusters)]

        for e, i in enumerate(self.clusters):
            value = self.dataset.critic_values[e].astype(np.float64)
            cluster_value[i].append(value)

        means = []
        stdevs = []

        for i in range(self.num_clusters):
            means.append(statistics.mean(cluster_value[i]))

            try:
                stdevs.append(statistics.stdev(cluster_value[i]))
            except Exception:
                stdevs.append(0)

        title = "Cluster Value Analysis"

        handles = [Patch(color="g"), Patch(color="r"), Patch(color="k")]
        labels = ["Initial", "Terminal", "Intermediate"]
        leg_title = "Cluster Stage"
        legend = {"handles": handles, "labels": labels, "title": leg_title}

        cluster_reward_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            legend=legend,
            error_bars=stdevs,
            xlabel="Cluster ID",
            ylabel="Mean Value",
            showall=True,
        )

        return cluster_reward_data
