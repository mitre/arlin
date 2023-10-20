from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Patch
from prettytable import PrettyTable

from arlin.analysis.visualization.colors import COLORS
from arlin.dataset.xrl_dataset import XRLDataset


class SAMDP:
    """Class for an SAMDP of an RL policy."""

    def __init__(self, clusters: np.ndarray, dataset: XRLDataset):
        """Intialize an SAMDP object.

        Args:
            clusters (np.ndarray): Generated cluster data.
            dataset (XRLDataset): XRLDataset from an RL policy.
        """
        self.clusters = clusters
        self.dataset = dataset

        self.samdp = self._generate()
        self.graph = self._generate_graph()

    def _generate(self) -> np.ndarray:
        """Generate an SAMDP.

        Returns:
            np.ndarray: Numpy array representation of the SAMDP.
        """
        logging.info("Generating SAMDP.")
        self.num_actions = len(np.unique(self.dataset.actions))
        self.num_clusters = len(np.unique(self.clusters))

        samdp_counts = np.zeros([self.num_clusters, self.num_actions, self.num_clusters])

        for i in range(len(self.clusters) - 1):
            terminated = self.dataset.terminateds[i]

            if not terminated:
                cur_cluster = self.clusters[i]
                action = self.dataset.actions[i]
                next_cluster = self.clusters[i + 1]

                if not cur_cluster == next_cluster:
                    samdp_counts[cur_cluster, action, next_cluster] += 1

        np.set_printoptions(suppress=True)
        np.seterr(divide="ignore", invalid="ignore")

        # Get the total number of out_edges for each cluster by action
        out_edges_per_action = np.sum(samdp_counts, axis=-1)

        # Add total row
        total_out_edges = np.expand_dims(np.sum(out_edges_per_action, 1), axis=-1)
        out_edges_per_action = np.append(out_edges_per_action, total_out_edges, axis=-1)

        out_edges_per_action = np.expand_dims(out_edges_per_action, axis=-1)
        # Expand total out_edges count to match size of samdp_counts
        expanded_out_edges = np.repeat(out_edges_per_action, self.num_clusters, axis=-1)

        # Get total out_edges to each cluster (not grouped by action)
        total_by_cluster = np.expand_dims(np.sum(samdp_counts, axis=1), 1)

        # Add the total counts row to the samdp_counts
        samdp_counts = np.concatenate([samdp_counts, total_by_cluster], axis=1)
        self.samdp_counts = samdp_counts

        samdp = samdp_counts / expanded_out_edges
        samdp = np.nan_to_num(samdp, nan=0)

        return samdp[:, :, :]

    def save_txt(self, save_dir: str) -> None:
        """Create a text table representation of the SAMDP.

        Args:
            save_dir (str): Dir to save the text SAMDP to.
        """
        samdp_data = ["SAMDP"]
        for from_cluster_id in range(self.num_clusters):
            table = PrettyTable()
            table.title = f"Cluster {from_cluster_id}"

            headers = [f"Cluster {i}" for i in range(self.num_clusters)]
            table.field_names = ["Action Value"] + headers
            for action in range(self.num_actions + 1):
                if action < self.num_actions:
                    row = [f"Action {action}"]
                else:
                    row = ["Total"]

                for to_cluster_id in range(self.num_clusters):
                    value = self.samdp_counts[from_cluster_id, action, to_cluster_id]
                    percent = self.samdp[from_cluster_id, action, to_cluster_id]
                    row.append(f"{value} | {round(percent*100, 2)}%")
                table.add_row(row)

            samdp_data.append(str(table))

        samdp_data = "\n".join(samdp_data)

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "samdp.txt"), "w") as f:
            f.write(samdp_data)

    def _generate_graph(self) -> nx.Graph:
        """Create a graph of this dataset's SAMDP using NetworkX.

        Each node represents a cluster from self.dataset["clusters"] and the edges
        represent the paths the agent takes in the dataset between clusters. An edge is
        added for each action taken that brings the agent from one cluster to another.
        For each action from a cluster, only the edge with the highest probability is
        shown, meaning there are other clusters that action can move the agent to but
        only the highest probability edge is shown.

        Returns:
            nx.Graph: NetworkX Graph representation of the SAMDP
        """

        logging.info("Generating SAMDP Graph.")

        G = nx.MultiDiGraph()

        G.add_nodes_from([f"Cluster {i}" for i in range(self.num_clusters)])

        for from_cluster_id in range(self.num_clusters):
            from_cluster = f"Cluster {from_cluster_id}"
            for action_id in range(self.num_actions):
                for to_cluster_id in range(self.num_clusters):
                    to_cluster = f"Cluster {to_cluster_id}"

                    prob = self.samdp[from_cluster_id, action_id, to_cluster_id]

                    if not prob == 0 and not from_cluster_id == to_cluster_id:
                        G.add_edge(
                            from_cluster,
                            to_cluster,
                            weight=prob,
                            action=action_id,
                            color=COLORS[action_id],
                        )

        self.graph = G
        self._set_node_attributes(self.graph)
        return self.graph

    def _generate_simplified_graph(self) -> nx.Graph:
        """Generate a simplified version of the SAMDP.

        In this graph, specific actions are not shown in the connections between nodes.
        Instead, a black line shows the connections between nodes.

        Returns:
            nx.Graph: Simplified version of the SAMDP with less informative connections.
        """
        G = nx.MultiDiGraph()
        G.add_nodes_from([f"Cluster {i}" for i in range(self.num_clusters)])

        for from_cluster_id in range(self.num_clusters):
            from_cluster = f"Cluster {from_cluster_id}"
            for to_cluster_id in range(self.num_clusters):
                to_cluster = f"Cluster {to_cluster_id}"

                prob = np.sum(self.samdp[from_cluster_id, -1, to_cluster_id])
                if not prob == 0 and not from_cluster_id == to_cluster_id:
                    G.add_edge(
                        from_cluster,
                        to_cluster,
                        weight=prob,
                        action=-1,
                        color="#000000",
                    )

        self._set_node_attributes(G)
        return G

    def _set_node_attributes(self, graph: nx.Graph):
        """Set the attributes of each node in the graph.

        Args:
            graph (nx.Graph): Graph object
        """
        self._set_node_colors(graph)
        self._set_node_edges(graph)

    def _set_node_colors(self, graph: nx.Graph):
        """Set the colors of each node in the graph.

        Args:
            graph (nx.Graph): Graph object
        """
        node_colors = {}
        for i in range(self.num_clusters):
            node_colors[f"Cluster {i}"] = COLORS[i]
        nx.set_node_attributes(graph, node_colors, "color")

    def _set_node_edges(self, graph: nx.Graph):
        """Set the colors for the edges of each node in the graph.

        Initial nodes have a green border, intermediate have a black border, and terminal
        have a red border.

        Args:
            graph (nx.Graph): Graph object
        """
        start_clusters = set(self.clusters[self.dataset.start_indices])
        term_clusters = set(self.clusters[self.dataset.term_indices])

        node_edges = {}
        for node_id in range(self.num_clusters):
            node_name = f"Cluster {node_id}"
            if node_id in start_clusters:
                node_edges[node_name] = "g"
            elif node_id in term_clusters:
                node_edges[node_name] = "r"
            else:
                node_edges[node_name] = "k"

        nx.set_node_attributes(graph, node_edges, "edge_color")

    def _generate_bfs_pos(self) -> Dict[nx.Graph.node, Tuple[int, int]]:
        """Generate the positioning for each node in the graph by breadth first search.

        Initial nodes are on the left, termainl nodes on the right, and intermediate
        nodes in between.

        Returns:
            Dict[Node, Tuple[int, int]]: Positions for each node in the graph.
        """
        pos = {}
        start_clusters = set(self.clusters[self.dataset.start_indices])
        term_clusters = set(self.clusters[self.dataset.term_indices])
        initial_nodes = [f"Cluster {i}" for i in start_clusters]
        terminal_nodes = [f"Cluster {i}" for i in term_clusters]

        bfs_layers = list(nx.bfs_layers(self.graph, initial_nodes))

        layers = []
        for i, layer_list in enumerate(bfs_layers):
            layers.append([])
            for j, node in enumerate(layer_list):
                if node in terminal_nodes:
                    pass
                pos[node] = (i, j)
                layers[i].append(node)

        depth = len(bfs_layers)
        for e, node in enumerate(terminal_nodes):
            pos[node] = (depth, e)

        return pos

    def _generate_edge_arcs(self, pos, edges: List) -> List[float]:
        """Generate the arcs for the connections between nodes.

        Connections have arcs if they overlap with other connections or go through
        nodes.

        Args:
            pos (Dict[Node, Tuple[int, int]]): Positions of each node in the graph.
            edges (List): List of edges

        Returns:
            List[float]: List of edge arcs.
        """
        edge_arcs = []
        for edge in edges:
            from_node_x, from_node_y = pos[edge[0]]
            to_node_x, to_node_y = pos[edge[1]]

            reverse_edge = self.graph.has_edge(edge[1], edge[0])

            arc = edge[2]
            if (from_node_x == to_node_x or from_node_y == to_node_y) or reverse_edge:
                arc += 1

            edge_arcs.append(0.05 * arc)

        return edge_arcs

    def save_complete_graph(self, file_path: str) -> nx.Graph:
        """Save the complete SAMDP as a matplotlib graph.

        Args:
            file_path (str): Path to save the graph image to.

        Returns:
            nx.Graph: Complete SAMDP graph
        """
        _ = plt.figure(figsize=(40, 20))
        plt.title("Complete SAMDP")

        pos = self._generate_bfs_pos()
        edge_arcs = self._generate_edge_arcs(pos, self.graph.edges(keys=True))

        colors = [node[1]["color"] for node in self.graph.nodes(data=True)]
        node_edges = [node[1]["edge_color"] for node in self.graph.nodes(data=True)]

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=4100,
            node_color=colors,
            edgecolors=node_edges,
            linewidths=5,
        )

        nx.draw_networkx_labels(self.graph, pos, font_color="whitesmoke")

        for i, edge in enumerate(self.graph.edges(data=True)):
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=[edge],
                connectionstyle=f"arc3,rad={edge_arcs[i]}",
                edge_color=edge[2]["color"],
                alpha=max(0, min(edge[2]["weight"] + 0.1, 1)),
                node_size=4000,
                arrowsize=25,
            )

        handles = [Patch(color=COLORS[i]) for i in range(self.num_actions)]
        labels = [f"Action {i}" for i in range(self.num_actions)]
        leg_title = "Actions"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        legend.update({"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"})
        plt.legend(**legend)

        plt.tight_layout()
        logging.info(f"Saving complete SAMDP graph png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()

        return self.graph

    def save_simplified_graph(self, file_path: str) -> nx.Graph:
        """Save a simplified version of the SAMDP graph.

        Edges do not include information about the action taken.

        Args:
            file_path (str): Path to save the SAMDP graph to.

        Returns:
            nx.Graph: Simplified SAMDP graph
        """
        _ = plt.figure(figsize=(40, 20))
        plt.title("Simplified SAMDP")

        G = self._generate_simplified_graph()
        pos = self._generate_bfs_pos()

        colors = [node[1]["color"] for node in self.graph.nodes(data=True)]
        node_edges = [node[1]["edge_color"] for node in self.graph.nodes(data=True)]

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=4100,
            node_color=colors,
            edgecolors=node_edges,
            linewidths=5,
        )

        nx.draw_networkx_labels(G, pos, font_color="whitesmoke")

        edges = G.edges(data=True, keys=True)
        edge_arcs = self._generate_edge_arcs(pos, edges)

        for i, edge in enumerate(edges):
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=[edge],
                connectionstyle=f"arc3,rad={edge_arcs[i]}",
                edge_color=edge[3]["color"],
                alpha=max(0, min(edge[3]["weight"] + 0.1, 1)),
                node_size=4000,
                arrowsize=25,
            )

        plt.tight_layout()
        logging.info(f"Saving simplified SAMDP graph png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()

        return G

    def save_likely_graph(self, file_path: str) -> nx.Graph:
        """Save a graph where only the most likely edges are shown.

        Args:
            file_path (str): Path to save graph image to.

        Returns:
            nx.Graph: Graph object with only most likely edges
        """
        _ = plt.figure(figsize=(40, 20))
        plt.title("Most Probable SAMDP")

        pos = self._generate_bfs_pos()

        colors = [node[1]["color"] for node in self.graph.nodes(data=True)]
        node_edges = [node[1]["edge_color"] for node in self.graph.nodes(data=True)]

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=4100,
            node_color=colors,
            edgecolors=node_edges,
            linewidths=5,
        )

        nx.draw_networkx_labels(self.graph, pos, font_color="whitesmoke")

        edges = []
        for node in self.graph.nodes():
            out_edges = self.graph.out_edges(node, data=True, keys=True)

            for action in range(self.num_actions):
                action_edges = [i for i in out_edges if i[3]["action"] == action]
                if not action_edges == []:
                    best_edge = sorted(
                        action_edges, key=lambda x: (x[3]["weight"], x[2]), reverse=True
                    )[0]
                    edges.append(best_edge)

        edge_arcs = self._generate_edge_arcs(pos, edges)

        for i, edge in enumerate(edges):
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=[edge],
                connectionstyle=f"arc3,rad={edge_arcs[i]}",
                edge_color=edge[3]["color"],
                alpha=max(0, min(edge[3]["weight"] + 0.1, 1)),
                node_size=4000,
                arrowsize=25,
            )

        handles = [Patch(color=COLORS[i]) for i in range(self.num_actions)]
        labels = [f"Action {i}" for i in range(self.num_actions)]
        leg_title = "Actions"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        legend.update({"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"})
        plt.legend(**legend)

        plt.tight_layout()
        logging.info(f"Saving most probable SAMDP graph png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()

        graph = nx.MultiDiGraph()
        graph.add_nodes_from(self.graph.nodes(data=True))
        graph.add_edges_from(edges)

        return graph

    def _find_best_path(
        self,
        from_cluster: str,
        to_cluster: str,
        paths: List[List[Tuple[str, str, int, Dict[str, Any]]]],
    ) -> Tuple[Dict[int, float], List]:
        """Calculate the probability of each path being taken.

        Args:
            from_cluster (str): Cluster to move from
            to_cluster (str): Cluster to move to
            paths (List[List[Tuple[str, str, int, Dict[str, Any]]]]): All simple paths
            from one cluster to another.

        Returns:
            Dict[int, float], List: Dictionary with actions as keys and highest
            probability to reach target from current node, List of edges that make up the
            most probably path between clusters
        """
        if len(paths) == 0:
            logging.info(f"\tNo paths found from {from_cluster} to {to_cluster}.")
            return []

        probs = {}
        best_paths = {}
        for path in paths:
            prob = 1
            action = path[0][3]["action"]
            for e, edge in enumerate(path):
                if e == 0:
                    edge_prob = edge[3]["weight"]
                else:
                    from_node = int(edge[0].split(" ")[-1])
                    to_node = int(edge[1].split(" ")[-1])
                    edge_prob = np.sum(self.samdp[from_node, -1, to_node])
                prob = prob * edge_prob

            if action in probs:
                if prob > probs[action]:
                    probs[action] = prob
                    best_paths[action] = path
            else:
                probs[action] = prob
                best_paths[action] = path

        logging.info(
            f"Highest probability of getting from {from_cluster} to {to_cluster}:"
        )
        for action in probs:
            logging.info(f"\tvia Action {action}: {round(probs[action] * 100, 2)}%")
            for i, edge in enumerate(best_paths[action]):
                if i == 0:
                    weight = round(edge[3]["weight"] * 100, 2)
                else:
                    from_id = int(edge[0].split(" ")[-1])
                    to_id = int(edge[1].split(" ")[-1])
                    weight = round(self.samdp[from_id, -1, to_id] * 100, 2)
                logging.info(f"\t\t{edge[0]} to {edge[1]} with {weight}%")

        best_action = max(probs, key=probs.get)
        logging.info(
            f"\tBest Option: Action {best_action} with "
            + f"{round(probs[best_action] * 100, 2)}%"
        )
        logging.info("\tBest Path:")
        for i, edge in enumerate(best_paths[best_action]):
            if i == 0:
                weight = round(edge[3]["weight"] * 100, 2)
            else:
                from_id = int(edge[0].split(" ")[-1])
                to_id = int(edge[1].split(" ")[-1])
                weight = round(self.samdp[from_id, -1, to_id] * 100, 2)
            logging.info(f"\t\t{edge[0]} to {edge[1]} with {weight}%")

        return best_paths[best_action]

    def save_paths(
        self,
        from_cluster_id: int,
        to_cluster_id: int,
        file_path: str,
        best_path_only: bool = False,
        verbose=False,
    ):
        """Save all paths from one cluster to another.

        Args:
            from_cluster_id (int): Cluster to move from
            to_cluster_id (int): Cluster to move to
            file_path (str): Path to save image to
            best_path_only (bool, optional): Do we only want to show the best path.
                Defaults to False.
            verbose (bool, optional): Do we want to show the complete edges instead of the
                simplified. Defaults to False.
        """
        from_cluster = f"Cluster {from_cluster_id}"
        to_cluster = f"Cluster {to_cluster_id}"

        if from_cluster not in self.graph.nodes():
            logging.warning(f"{from_cluster} is not a valid cluster.")
            return

        if to_cluster not in self.graph.nodes():
            logging.warning(f"{to_cluster} is not a valid cluster.")
            return

        _ = plt.figure(figsize=(40, 20))
        plt.title(f"SAMDP Paths from {from_cluster} to {to_cluster}")

        logging.info(f"Finding paths from {from_cluster} to {to_cluster}...")

        if verbose:
            graph = copy.deepcopy(self.graph)
        else:
            graph = self._generate_simplified_graph()

            out_edges = graph.out_edges(from_cluster)
            graph.remove_edges_from(list(out_edges))

            action_out_edges = self.graph.out_edges(from_cluster, data=True, keys=True)

            graph.add_edges_from(list(action_out_edges))

        paths = list(nx.all_simple_edge_paths(graph, from_cluster, to_cluster))

        if len(paths) == 0:
            logging.info(f"\tNo paths found from {from_cluster} to {to_cluster}.")
            plt.close()
            return

        updated_paths = []
        full_edge_list = []
        edge_list = []
        for path in paths:
            data_path = []
            for edge in path:
                edge_data = graph.get_edge_data(edge[0], edge[1], edge[2])
                updated_edge = (edge[0], edge[1], edge[2], edge_data)
                data_path.append(updated_edge)

                if updated_edge not in full_edge_list:
                    full_edge_list.append(updated_edge)
                    edge_list.append(edge)

            updated_paths.append(data_path)

        best_path = self._find_best_path(from_cluster, to_cluster, updated_paths)

        if best_path_only:
            full_edge_list = best_path
            edge_list = []
            for edge in best_path:
                edge_list.append((edge[0], edge[1], edge[2]))

        subgraph = nx.edge_subgraph(graph, edge_list)

        pos = self._generate_bfs_pos()
        edge_arcs = self._generate_edge_arcs(pos, edge_list)

        colors = [node[1]["color"] for node in subgraph.nodes(data=True)]
        node_edges = [node[1]["edge_color"] for node in subgraph.nodes(data=True)]

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=4100,
            node_color=colors,
            edgecolors=node_edges,
            linewidths=5,
        )

        nx.draw_networkx_labels(subgraph, pos, font_color="whitesmoke")

        for i, edge in enumerate(full_edge_list):
            nx.draw_networkx_edges(
                subgraph,
                pos,
                edgelist=[edge],
                connectionstyle=f"arc3,rad={edge_arcs[i]}",
                edge_color=edge[3]["color"],
                alpha=max(0, min(edge[3]["weight"] + 0.1, 1)),
                node_size=4000,
                arrowsize=25,
            )

        handles = [Patch(color=COLORS[i]) for i in range(self.num_actions)]
        labels = [f"Action {i}" for i in range(self.num_actions)]
        leg_title = "Actions"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        legend.update({"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"})
        plt.legend(**legend)

        plt.tight_layout()
        logging.info(
            f"Saving SAMDP path from {from_cluster} to {to_cluster} png to {file_path}..."
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()

    def save_terminal_paths(
        self,
        file_path: str,
        best_path: bool = False,
        term_cluster_id: Optional[int] = None,
    ):
        """Save all paths into all terminal nodes.

        Args:
            file_path (str): Path to save image to
            best_path (bool, optional): Do we only want to show the best paths between
                nodes. Defaults to False.
            term_cluster_id (Optional[int], optional): Cluster ID that we want to limit
                paths to instead of all paths. Defaults to None.
        """
        graph = copy.deepcopy(self.graph)

        term_nodes = []
        for node in graph.nodes(data=True):
            if node[1]["edge_color"] == "r":
                term_nodes.append(node[0])

        if term_cluster_id is not None:
            cluster_node = f"Cluster {term_cluster_id}"

            if cluster_node not in term_nodes:
                logging.info(f"Cluster {term_cluster_id} is not a terminal cluster.")
                return

            term_nodes = [cluster_node]

        _ = plt.figure(figsize=(40, 20))
        plt.title(f"All SAMDP connections to terminal cluster {term_cluster_id}")
        logging.info(f"Finding connections to terminal cluster {term_cluster_id}...")

        edge_list = []
        full_edge_list = []
        for node in term_nodes:
            full_in_edges = graph.in_edges(node, data=True, keys=True)

            if not best_path:
                for edge in full_in_edges:
                    full_edge_list.append(edge)
                    edge_list.append((edge[0], edge[1], edge[2]))
            else:
                node_dict = {}
                for edge in full_in_edges:
                    if edge[0] not in node_dict.keys():
                        node_dict[edge[0]] = edge
                    else:
                        if node_dict[edge[0]][3]["weight"] < edge[3]["weight"]:
                            node_dict[edge[0]] = edge

                for edge in node_dict.values():
                    full_edge_list.append(edge)
                    edge_list.append((edge[0], edge[1], edge[2]))

        subgraph = self.graph.edge_subgraph(edge_list)

        pos = self._generate_bfs_pos()
        edge_arcs = self._generate_edge_arcs(pos, edge_list)

        colors = [node[1]["color"] for node in subgraph.nodes(data=True)]
        node_edges = [node[1]["edge_color"] for node in subgraph.nodes(data=True)]

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=4100,
            node_color=colors,
            edgecolors=node_edges,
            linewidths=5,
        )

        nx.draw_networkx_labels(subgraph, pos, font_color="whitesmoke")

        for i, edge in enumerate(full_edge_list):
            nx.draw_networkx_edges(
                subgraph,
                pos,
                edgelist=[edge],
                connectionstyle=f"arc3,rad={edge_arcs[i]}",
                edge_color=edge[3]["color"],
                node_size=4000,
                arrowsize=25,
            )

        handles = [Patch(color=COLORS[i]) for i in range(self.num_actions)]
        labels = [f"Action {i}" for i in range(self.num_actions)]
        leg_title = "Actions"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        legend.update({"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"})
        plt.legend(**legend)

        plt.tight_layout()
        logging.info(f"Saving all SAMDP paths to terminal clusters png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()

    def save_all_paths_to(
        self, to_cluster_id: int, file_path: str, verbose: bool = False
    ):
        """Save all possible paths from an initial node to given node.

        Args:
            to_cluster_id (int): Cluster we want to get to
            file_path (str): Path to save image to
            verbose (bool, optional): Do we want to show complete graph edges instead of
                simplified. Defaults to False.
        """
        to_cluster = f"Cluster {to_cluster_id}"

        if to_cluster not in self.graph.nodes():
            logging.warning(f"{to_cluster} is not a valid cluster.")
            return

        _ = plt.figure(figsize=(40, 20))
        plt.title(f"All SAMDP Paths to {to_cluster}")

        logging.info(f"Finding paths to {to_cluster}...")

        if verbose:
            graph = copy.deepcopy(self.graph)
        else:
            graph = self._generate_simplified_graph()

            in_edges = graph.in_edges(to_cluster)
            graph.remove_edges_from(list(in_edges))

            action_in_edges = self.graph.in_edges(to_cluster, data=True, keys=True)

            graph.add_edges_from(list(action_in_edges))

        paths = []

        for node in graph.nodes():
            if node == to_cluster:
                continue

            paths += list(nx.all_simple_edge_paths(graph, node, to_cluster))

        if len(paths) == 0:
            logging.info(f"\tNo paths found to {to_cluster}.")
            plt.close()
            return

        updated_paths = []
        full_edge_list = []
        edge_list = []
        for path in paths:
            data_path = []
            for edge in path:
                edge_data = graph.get_edge_data(edge[0], edge[1], edge[2])
                updated_edge = (edge[0], edge[1], edge[2], edge_data)
                data_path.append(updated_edge)

                if updated_edge not in full_edge_list:
                    full_edge_list.append(updated_edge)
                    edge_list.append(edge)

            updated_paths.append(data_path)

        subgraph = nx.edge_subgraph(graph, edge_list)

        pos = self._generate_bfs_pos()
        edge_arcs = self._generate_edge_arcs(pos, edge_list)

        colors = [node[1]["color"] for node in subgraph.nodes(data=True)]
        node_edges = [node[1]["edge_color"] for node in subgraph.nodes(data=True)]

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=4100,
            node_color=colors,
            edgecolors=node_edges,
            linewidths=5,
        )

        nx.draw_networkx_labels(subgraph, pos, font_color="whitesmoke")

        for i, edge in enumerate(full_edge_list):
            nx.draw_networkx_edges(
                subgraph,
                pos,
                edgelist=[edge],
                connectionstyle=f"arc3,rad={edge_arcs[i]}",
                edge_color=edge[3]["color"],
                alpha=max(0, min(edge[3]["weight"] + 0.1, 1)),
                node_size=4000,
                arrowsize=25,
            )

        handles = [Patch(color=COLORS[i]) for i in range(self.num_actions)]
        labels = [f"Action {i}" for i in range(self.num_actions)]
        leg_title = "Actions"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        legend.update({"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"})
        plt.legend(**legend)

        plt.tight_layout()
        logging.info(f"Saving all SAMDP paths to {to_cluster} png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()
