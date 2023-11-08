import numpy as np
from matplotlib.patches import Patch

from arlin.analysis.visualization import COLORS, GraphData
from arlin.dataset import XRLDataset


class LatentAnalyzer:
    """Class to analyze latent embeddings and generate data to visualize."""

    def __init__(self, embeddings: np.ndarray, dataset: XRLDataset):
        """Initialize an instance of a LatentAnalyzer

        Args:
            embeddings (np.ndarray): Generated embeddings
            dataset (XRLDataset): XRLDataset created from an RL policy
        """
        self.embeddings = embeddings
        self.dataset = dataset
        self.num_embeddings = len(self.embeddings)
        self.x = embeddings[:, 0]
        self.y = embeddings[:, 1]

    def embeddings_graph_data(self) -> GraphData:
        """
        Generate data necessary for creating embedding graphs.

        Returns:
            GraphData: Data to visualize
        """

        colors = ["#5A5A5A"] * self.num_embeddings
        title = "Embeddings"

        embed_data = GraphData(x=self.x, y=self.y, title=title, colors=colors)

        return embed_data

    def clusters_graph_data(self, clusters: np.ndarray) -> GraphData:
        """
        Generate data necessary for creating cluster graphs.

        Returns:
            GraphData: Data to visualize
        """
        num_clusters = len(np.unique(clusters))
        colors = [COLORS[i] for i in clusters]
        title = f"{num_clusters} Clusters"

        handles = [Patch(color=COLORS[i], label=str(i)) for i in range(num_clusters)]
        labels = [f"Cluster {i}" for i in range(num_clusters)]
        leg_title = "Cluster Groups"
        legend = {"handles": handles, "labels": labels, "title": leg_title}

        cluster_data = GraphData(
            x=self.x, y=self.y, title=title, colors=colors, legend=legend
        )

        return cluster_data

    def decision_boundary_graph_data(self) -> GraphData:
        """
        Generate data necessary for creating decision boundary graphs.

        Returns:
            GraphData: Data to visualize
        """
        colors = [COLORS[i] for i in self.dataset.actions]
        title = "Decision Boundaries for Taken Actions"

        num_actions = len(np.unique(self.dataset.actions))
        handles = [Patch(color=COLORS[i], label=str(i)) for i in range(num_actions)]
        labels = [f"{i}" for i in range(num_actions)]
        leg_title = "Action Values"
        legend = {"handles": handles, "labels": labels, "title": leg_title}

        decision_boundary_data = GraphData(
            x=self.x, y=self.y, title=title, colors=colors, legend=legend
        )

        return decision_boundary_data

    def episode_prog_graph_data(self) -> GraphData:
        """
        Generate data necessary for creating episode progression graphs.

        Returns:
            GraphData: Data to visualize
        """

        colors = self.dataset.steps
        title = "Episode Progression"

        episode_prog_data = GraphData(
            x=self.x, y=self.y, title=title, colors=colors, cmap="viridis"
        )

        return episode_prog_data

    def confidence_data(self) -> GraphData:
        """
        Generate data necessary for creating episode progression graphs.

        Returns:
            GraphData: Data to visualize
        """

        try:
            colors = self.dataset.dist_probs
        except Exception:
            error_str = (
                "Current dataset does not include 'dist_probs' attribute."
                "Confidence data is only available for PPO datasets."
            )
            raise ValueError(error_str)

        colors = np.amax(self.dataset.dist_probs, axis=1)
        title = "Policy Confidence in Greedy Action"

        conf_data = GraphData(
            x=self.x, y=self.y, title=title, colors=colors, cmap="RdYlGn"
        )

        return conf_data

    def initial_terminal_state_data(self) -> GraphData:
        """Generate data necessary for creating initial/terminal state graphs.

        Returns:
            GraphData: Data to visualize
        """

        colors = [COLORS[0] if i else "#F5F5F5" for i in self.dataset.terminateds]
        for i in self.dataset.start_indices:
            colors[i] = COLORS[1]
        title = "Initial and Terminal States"

        handles = [Patch(color=COLORS[1]), Patch(color=COLORS[0])]
        labels = ["Initial", "Terminal"]
        leg_title = "State Type"

        legend = {"handles": handles, "labels": labels, "title": leg_title}

        state_data = GraphData(
            x=self.x, y=self.y, title=title, colors=colors, legend=legend
        )

        return state_data
