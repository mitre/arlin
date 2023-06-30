from __future__ import annotations
import numpy as np
import networkx as nx
import logging
import os
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from arlin.data_analysis.graphers.colors import COLORS

from typing import Dict, Tuple, List, Optional, Any

from arlin.data_analysis.xrl_dataset import XRLDataset

class SAMDP():
    
    def __init__(self,
                 clusters: np.ndarray,
                 dataset: XRLDataset
                 ):
        
        self.clusters = clusters
        self.dataset = dataset
        
        self.samdp = self._generate()
        self.graph = self._generate_graph()
    
    def _generate(self) -> SAMDP:
        
        logging.info("Generating SAMDP.")
        self.num_actions = len(np.unique(self.dataset.actions))
        self.num_clusters = len(np.unique(self.clusters))
        
        samdp_counts = np.zeros([self.num_clusters, 
                                 self.num_actions, 
                                 self.num_clusters])
        
        for i in range(len(self.clusters) - 1):
            done = self.dataset.dones[i]
            
            if not done:
                cur_cluster = self.clusters[i]
                action = self.dataset.actions[i]
                next_cluster = self.clusters[i+1]
                
                if not cur_cluster == next_cluster:
                    samdp_counts[cur_cluster, action, next_cluster] += 1
        
        self.samdp_counts = samdp_counts
        
        np.set_printoptions(suppress=True)
        np.seterr(divide='ignore', invalid='ignore')
        
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
        
        samdp = samdp_counts / expanded_out_edges
        samdp = np.nan_to_num(samdp, nan=0)

        return samdp

    def save_txt(self, file_path: str) -> None:
        """Create a txt table of the SAMDP.

        Args:
            samdp_counts (np.ndarray): The number of times an agent moved from one
                cluster to another along with the action taken to get to the latter.
        """
        samdp_data = ["SAMDP"]
        for from_cluster_id in range(self.num_clusters):
            table = PrettyTable()
            table.title = f"Cluster {from_cluster_id}"
            
            headers = [f"Cluster {i}" for i in range(self.num_clusters)]
            table.field_names = ["Action Value"] + headers
            for action in range(self.num_actions):
                row = [f'Action {action}']
                
                for to_cluster_id in range(self.num_clusters):
                    value = self.samdp_counts[from_cluster_id, action, to_cluster_id]
                    percent = self.samdp[from_cluster_id, action, to_cluster_id]
                    row.append(f"{value} | {round(percent*100, 2)}%")
                table.add_row(row)
            
            samdp_data.append(str(table))
        
        samdp_data = "\n".join(samdp_data)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
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
                        G.add_edge(from_cluster, 
                           to_cluster, 
                           weight=prob,
                           action=action_id,
                           color=COLORS[action_id])
        
        self.graph = G
        self._set_node_attributes(self.graph)
        return self.graph

    def _generate_simplified_graph(self) -> nx.Graph:
        G = nx.MultiDiGraph()
        G.add_nodes_from([f"Cluster {i}" for i in range(self.num_clusters)])
        
        for from_cluster_id in range(self.num_clusters):
            from_cluster = f"Cluster {from_cluster_id}"
            for to_cluster_id in range(self.num_clusters):
                to_cluster = f"Cluster {to_cluster_id}"
                
                prob = self.samdp[from_cluster_id, self.num_actions, to_cluster_id]
                
                if not prob == 0 and not from_cluster_id == to_cluster_id:
                    G.add_edge(from_cluster, 
                        to_cluster, 
                        weight=prob,
                        action=-1,
                        color='#000000')
        
        self._set_node_attributes(G)
        return G
    
    def _set_node_attributes(self, graph: nx.Graph):
        self._set_node_colors(graph)
        self._set_node_edges(graph)
    
    def _set_node_colors(self, graph: nx.Graph):
        node_colors = {}
        for i in range(self.num_clusters):
            node_colors[f'Cluster {i}'] = COLORS[i]
        nx.set_node_attributes(graph, node_colors, 'color')
    
    def _set_node_edges(self, graph: nx.Graph):
        start_clusters = set(self.clusters[self.dataset.start_indices])
        done_clusters = set(self.clusters[self.dataset.done_indices])
        
        node_edges = {}
        for node_id in range(self.num_clusters):
            node_name = f'Cluster {node_id}'
            if node_id in start_clusters and node_id in done_clusters:
                node_edges[node_name] = 'y'
            elif node_id in start_clusters:
                node_edges[node_name] = 'g'
            elif node_id in done_clusters:
                node_edges[node_name] = 'r'
            else:
                node_edges[node_name] = 'k'
        
        nx.set_node_attributes(graph, node_edges, 'edge_color')
    
    def _generate_bfs_pos(self):

        pos = {}
        start_clusters = set(self.clusters[self.dataset.start_indices])
        done_clusters = set(self.clusters[self.dataset.done_indices])
        initial_nodes = [f'Cluster {i}' for i in start_clusters]
        terminal_nodes = [f'Cluster {i}' for i in done_clusters]
        
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
        layers.append([])
        for e, node in enumerate(terminal_nodes):
            pos[node] = (depth, e)
            layers[depth].append(node)
        
        return pos
    
    def _generate_edge_arcs(self, pos, edges: List):
    
        edge_arcs = []
        for edge in edges:              
            from_node_x, from_node_y = pos[edge[0]]
            to_node_x, to_node_y = pos[edge[1]]
            
            arc = edge[2]
            if from_node_x == to_node_x or from_node_y == to_node_y:
                arc += 1
            
            edge_arcs.append(0.05*arc)

        return edge_arcs
    
    def save_complete_graph(self, file_path: str):
        _ = plt.figure(figsize=(40,20))
        plt.title('Complete SAMDP')
        
        pos = self._generate_bfs_pos()
        edge_arcs = self._generate_edge_arcs(pos, self.graph.edges(keys=True))
        
        colors = [node[1]['color'] for node in self.graph.nodes(data=True)]
        node_edges = [node[1]['edge_color'] for node in self.graph.nodes(data=True)]
        
        nx.draw_networkx_nodes(self.graph, 
                               pos,
                               node_size=4100,
                               node_color=colors,
                               edgecolors=node_edges,
                               linewidths=5)
        
        nx.draw_networkx_labels(self.graph, pos, font_color='whitesmoke')

        for i, edge in enumerate(self.graph.edges(data=True)):
            nx.draw_networkx_edges(self.graph, 
                                   pos, 
                                   edgelist=[edge], 
                                   connectionstyle=f"arc3,rad={edge_arcs[i]}",
                                   edge_color=edge[2]['color'],
                                   alpha=edge[2]['weight'],
                                   node_size=4000, 
                                   arrowsize=25)
        
        plt.tight_layout()
        logging.info(f"Saving complete SAMDP graph png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()
        
        return self.graph
    
    def save_simplified_graph(self, file_path: str):
        _ = plt.figure(figsize=(40,20))
        plt.title('Simplified SAMDP')
        
        G = self._generate_simplified_graph()
        pos = self._generate_bfs_pos()
        
        colors = [node[1]['color'] for node in self.graph.nodes(data=True)]
        node_edges = [node[1]['edge_color'] for node in self.graph.nodes(data=True)]
        
        nx.draw_networkx_nodes(G, 
                               pos,
                               node_size=4100,
                               node_color=colors,
                               edgecolors=node_edges,
                               linewidths=5)
        
        nx.draw_networkx_labels(G, pos, font_color='whitesmoke')
        
        edges = G.edges(data=True, keys=True)
        edge_arcs = self._generate_edge_arcs(pos, edges)
        
        for i, edge in enumerate(edges):
            nx.draw_networkx_edges(self.graph, 
                                    pos, 
                                    edgelist=[edge], 
                                    connectionstyle=f"arc3,rad={edge_arcs[i]}",
                                    edge_color=edge[3]['color'],
                                    alpha=edge[3]['weight'],
                                    node_size=4000, 
                                    arrowsize=25)
        
        plt.tight_layout()
        logging.info(f"Saving simplified SAMDP graph png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()
        
        return G
    
    def save_early_termination_paths(self, file_path: str):
        _ = plt.figure(figsize=(15, 15))
        plt.title('SAMDP Early Termination Paths')
        
        start_clusters = list(set(self.clusters[self.dataset.start_indices]))
        done_clusters = list(set(self.clusters[self.dataset.done_indices]))
        initial_nodes = [f'Cluster {i}' for i in start_clusters]
        terminal_nodes = [f'Cluster {i}' for i in done_clusters]
        
        edges = []
        for edge in self.graph.edges:
            if edge[0] in initial_nodes and edge[1] in terminal_nodes:
                edges.append(edge)
        
        subgraph = nx.edge_subgraph(self.graph, edges)
        
        pos = {}
        initial = 0
        terminal = 0
        colors = []
        node_edges = []
        labels = {}
        for node in subgraph.nodes(data=True):
            if node[0] in initial_nodes:
                pos[node[0]] = (0, initial)
                initial += 1
            
            if node[0] in terminal_nodes:
                pos[node[0]] = (1, terminal)
                terminal += 1
            
            colors.append(node[1]['color'])
            node_edges.append(node[1]['edge_color'])
            labels[node[0]] = node[0]
        
        nx.draw_networkx_nodes(subgraph, 
                               pos,
                               node_size=4100,
                               node_color=colors,
                               edgecolors=node_edges,
                               linewidths=5)

        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_color='whitesmoke')
        
        for edge in subgraph.edges(data=True, keys=True):
            nx.draw_networkx_edges(subgraph, 
                                        pos, 
                                        edgelist=[edge], 
                                        connectionstyle=f"arc3,rad={0.1 * edge[2]}",
                                        edge_color=edge[3]['color'],
                                        alpha=edge[3]['weight'],
                                        node_size=4000, 
                                        arrowsize=25)
        plt.tight_layout()
        logging.info(f"Saving SAMDP early termination paths png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()
        
        return subgraph
    
    def save_likely_paths(self, file_path: str):
        _ = plt.figure(figsize=(40,20))
        plt.title('Most Probable SAMDP')
        
        pos = self._generate_bfs_pos()
        
        colors = [node[1]['color'] for node in self.graph.nodes(data=True)]
        node_edges = [node[1]['edge_color'] for node in self.graph.nodes(data=True)]
        
        nx.draw_networkx_nodes(self.graph, 
                               pos,
                               node_size=4100,
                               node_color=colors,
                               edgecolors=node_edges,
                               linewidths=5)
        
        nx.draw_networkx_labels(self.graph, pos, font_color='whitesmoke')

        edges = []
        for node in self.graph.nodes():
            out_edges = self.graph.out_edges(node, data=True, keys=True)
            
            for action in range(self.num_actions):
                action_edges = [i for i in out_edges if i[3]['action'] == action]
                if not action_edges == []:
                    best_edge = sorted(action_edges, key=lambda x: x[3]['weight'])[-1]
                    edges.append(best_edge)
        
        edge_arcs = self._generate_edge_arcs(pos, edges)
        
        for i, edge in enumerate(edges):
            nx.draw_networkx_edges(self.graph, 
                                   pos, 
                                   edgelist=[edge], 
                                   connectionstyle=f"arc3,rad={edge_arcs[i]}",
                                   edge_color=edge[3]['color'],
                                   alpha=edge[3]['weight'],
                                   node_size=4000, 
                                   arrowsize=25)
        
        plt.tight_layout()
        logging.info(f"Saving most probable SAMDP graph png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()
        
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(self.graph.nodes(data=True))
        graph.add_edges_from(edges)
        
        return graph
    
    
    def _calculate_path_probs(self,
                              from_cluster: str,
                              to_cluster: str,
                              paths: List[Tuple[str, str, int, Dict[str, Any]]]
                              ) -> Dict[int, float]:
        """Calculate the probability of each path being taken.

        Args:
            paths (List[Tuple[str, str, int, Dict[str, Any]]]): All simple paths from
            one cluster to another.

        Returns:
            Dict[int, float]: Dictionary with actions as keys and highest probability
                to reach target from current node.
        """
        
        if len(paths) == 0:
            logging.info(f"No paths found from {from_cluster} to {to_cluster}.")
            return {}, []
        
        probs = {}
        best_paths = {}
        for path in paths:
            prob = 1
            action = path[0][3]['action']
            for e, edge in enumerate(path):
                if e == 0:
                    edge_prob = edge[3]['weight']
                else:
                    from_node = int(edge[0].split(' ')[-1])
                    to_node = int(edge[1].split(' ')[-1])
                    edge_prob = self.samdp[from_node][-1][to_node]
                prob = prob * edge_prob
            
            if action in probs:
                if prob > probs[action]:
                    probs[action] = prob
                    best_paths[action] = path
            else:
                probs[action] = prob
                best_paths[action] = path

        logging.info(f"Highest probability of getting from {from_cluster} to {to_cluster}:")
        for action in probs:
            logging.info(f"\tvia Action {action}: {round(probs[action] * 100, 2)}%")
            for edge in best_paths[action]:
                logging.info(f"\t\t{edge[0]} to {edge[1]} with {round(edge[3]['weight'] * 100, 2)}%")
            
        best_action = max(probs, key=probs.get)
        logging.info(f'\tBest Option: Action {best_action} with {round(probs[best_action] * 100, 2)}%')
        logging.info(f'\tBest Path:')
        for edge in best_paths[best_action]:
            logging.info(f"\t\t{edge[0]} to {edge[1]} with {round(edge[3]['weight'] * 100, 2)}%")
        
        return probs, best_paths[best_action]
    
    def save_paths(self,
                   from_cluster_id: int, 
                   to_cluster_id: int,
                   file_path: str,
                   best_path_only: bool = False):
        """Find simple paths between two clusters within SAMDP.

        Args:
            from_cluster_id (int): ID of cluster to start from.
            to_cluster_id (int): ID of cluster to end at.
        """
        from_cluster = f'Cluster {from_cluster_id}'
        to_cluster = f'Cluster {to_cluster_id}'
        
        _ = plt.figure(figsize=(40,20))
        plt.title(f'SAMDP Paths from {from_cluster} to {to_cluster}')
        
        logging.info(f'Finding paths from {from_cluster} to {to_cluster}...')
        
        simplified_graph = self._generate_simplified_graph()
        
        out_edges = simplified_graph.out_edges(from_cluster)
        simplified_graph.remove_edges_from(list(out_edges))
        
        action_out_edges = self.graph.out_edges(from_cluster, data=True, keys=True)
        
        simplified_graph.add_edges_from(list(action_out_edges))
        
        paths = list(nx.all_simple_edge_paths(simplified_graph, 
                                              from_cluster, 
                                              to_cluster))
        
        updated_paths = []
        full_edge_list = []
        edge_list = []
        for path in paths:
            data_path = []
            for edge in path:
                edge_data = simplified_graph.get_edge_data(edge[0], edge[1], edge[2])
                updated_edge = (edge[0], edge[1], edge[2], edge_data)
                data_path.append(updated_edge)
                
                if updated_edge not in full_edge_list:
                    full_edge_list.append(updated_edge)
                    edge_list.append(edge)
            
            updated_paths.append(data_path)
        
        _, best_path = self._calculate_path_probs(from_cluster,
                                          to_cluster, 
                                          updated_paths)
        
        if best_path == []:
            return
        
        if best_path_only:
            full_edge_list = best_path
            edge_list = []
            for edge in best_path:
                edge_list.append((edge[0], edge[1], edge[2]))
        
        subgraph = nx.edge_subgraph(simplified_graph, edge_list)
        
        pos = self._generate_bfs_pos()
        edge_arcs = self._generate_edge_arcs(pos, edge_list)
        
        colors = [node[1]['color'] for node in subgraph.nodes(data=True)]
        node_edges = [node[1]['edge_color'] for node in subgraph.nodes(data=True)]
        
        nx.draw_networkx_nodes(subgraph, 
                               pos,
                               node_size=4100,
                               node_color=colors,
                               edgecolors=node_edges,
                               linewidths=5)
        
        nx.draw_networkx_labels(subgraph, pos, font_color='whitesmoke')
        
        for i, edge in enumerate(full_edge_list):
            nx.draw_networkx_edges(subgraph, 
                                   pos, 
                                   edgelist=[edge], 
                                   connectionstyle=f"arc3,rad={edge_arcs[i]}",
                                   edge_color=edge[3]['color'],
                                   alpha=edge[3]['weight'],
                                   node_size=4000, 
                                   arrowsize=25)
        
        plt.tight_layout()
        logging.info(f"Saving SAMDP path from {from_cluster} to {to_cluster} png to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, format="PNG")
        plt.close()
        