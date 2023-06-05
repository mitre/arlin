from typing import Any, Dict, List, Tuple
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import os
import pickle
import logging
import pathlib
import shutil
import networkx as nx
from prettytable import PrettyTable
import statistics
import itertools

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import arlin.utils.data_analysis_utils as utils
from arlin.data_analysis.graph_data import GraphData

class DataAnalyzer():
    """
    #### Description
    Class for analyzing, embedding, clustering, and visualizing RL dataset information.
    """
    
    np.random.seed(1234)
    
    def load_data(self, dataset_path: str) -> None:
        """
        Load the data from the dataset.
        
        Args:
            - dataset_path (str): Path to the dataset pickle file
        """
        self.dataset_dir = os.path.dirname(dataset_path)
        self.filename = os.path.basename(dataset_path).split('.')[0]
        
        dataset_file = open(dataset_path,'rb')
        self.dataset = pickle.load(dataset_file)
        self.num_datapoints = len(self.dataset['observations'])
        dataset_file.close()
        
        self._get_episode_data()
        self._get_distinct_state_data()
        
        self._setup_outputs()
        
    def _setup_outputs(self):
        """
        Create necessary directories for outputs and save a copy of the config.
        """
        
        if not os.path.exists('./outputs'):
            os.mkdir('./outputs')
            trial_name = 'trial_0'
        else:
            output_path = pathlib.Path('./outputs')
            
            highest = 0
            for item in output_path.iterdir():
                if item.is_dir():
                    name = os.path.basename(item)
                    split_name = name.split('_')
                    if split_name[0] == 'trial' and int(split_name[1]) > highest:
                        highest = int(split_name[1])
            
            trial_name = f"trial_{highest + 1}"
        
        os.mkdir(f'./outputs/{trial_name}')
        os.mkdir(f'./outputs/{trial_name}/individual_graphs/')
        os.mkdir(f'./outputs/{trial_name}/combo_graphs/')
        os.mkdir(f'./outputs/{trial_name}/SAMDP/')
        
        self.trial_path = f"./outputs/{trial_name}/"
        
        shutil.copy2('./config.yaml', f'./outputs/{trial_name}/config.yaml')
    
    def _get_episode_data(self):
        """
        Extract start and end data about the episodes from the dataset.
        """
        logging.info("Extracting episode data...")
        done_indices = np.where(self.dataset["dones"] == 1)[0]
        start_indices = done_indices + 1
        start_indices = np.insert(start_indices, 0, 0)
        if start_indices[-1] == len(self.dataset["dones"]):
            start_indices = start_indices[:-1]
        
        final_states = self.dataset["observations"][done_indices]
        start_states = self.dataset["observations"][start_indices]
        
        self.dataset["done_indices"] = done_indices
        self.dataset["start_indices"] = start_indices
        self.dataset["final_states"] = final_states
        self.dataset["start_states"] = start_states
    
    def _get_distinct_state_data(self):
        """
        Separate out distinct states from the given dataset states due to t-sne issues
        with duplicate states. Link together distinct states as the embeddings will be the
        same for all duplicates of a state.
        """
        logging.info("Extracting disctinct state data...")
        outputs = np.unique(
            self.dataset["observations"], 
            return_index=True, 
            return_inverse=True, 
            axis=0)
        
        unique_states, unique_indices, state_mapping = outputs
        logging.info(f"\tFound {unique_states.shape[0]} distinct states!")
        
        self.dataset['unique_states'] = unique_states
        self.dataset['unique_indices'] = unique_indices
        self.dataset['state_mapping'] = state_mapping
    
    def get_embeddings(
        self, 
        activation_key: str, 
        num_components: int,
        perplexity: int,
        n_iter: int,
        load_embeddings: bool = False
        ) -> np.ndarray:
        """
        Load pre-made embeddings or generate new embeddings from an activation key.
        
        Args:
            - activation_key (str): Data to perform dimensionality reduction on
            - num_components (int): Number of components to reduce to
            - perplexity (int): Value for balancing local vs global aspects
            - n_iter (int): Number of iterations during training for T-SNE
            - load_embeddings (bool): Wheter or not to load a premade embedding
            
        Returns:
            - np.ndarray: Embeddings of values from the activation key
        """
        
        embeddings_dir = os.path.join(self.dataset_dir, "embeddings",
                                      self.filename, activation_key)
        embed_filename = f"{perplexity}_perplexity-{n_iter}_iter-embeddings.pkl"
        
        self.perplexity = perplexity
        self.activation_key = activation_key
        self.n_iter = n_iter
        
        if load_embeddings:
            self.dataset['embeddings'] = utils.load_data(embeddings_dir, embed_filename)
        else:
            
            self.dataset['embeddings'] = self._generate_embeddings(
                activation_key, num_components, perplexity, n_iter
            )
            utils.save_data(self.dataset['embeddings'], embeddings_dir, embed_filename)
            
        return self.dataset['embeddings']
    
    def _generate_embeddings(
        self,
        activation_key: str, 
        num_components: int,
        perplexity: int,
        n_iter: int
        ) -> np.ndarray:
        """
        Generate new embeddings from an activation key.
        
        Args:
            - activation_key (str): Data to perform dimensionality reduction on
            - num_components (int): Number of components to reduce to
            - perplexity (int): Value for balancing local vs global aspects
            - n_iter (int): Number of iterations during training for T-SNE
        
        Returns:
            - np.ndarray: Embeddings of values from the activation key
        """
        
        embedder = TSNE(
            n_components=num_components, 
            perplexity=perplexity,
            n_iter=n_iter,
            verbose=1, 
            random_state=12345)
    
        activations = self.dataset[activation_key]
        unique_activations = activations[self.dataset["unique_indices"]]
        
        embeddings = embedder.fit_transform(unique_activations)
        embeddings = [embeddings[index] for index in self.dataset["state_mapping"]]
        embeddings = np.array(embeddings)
        
        return embeddings
    
    def get_clusters(
        self,
        num_clusters: int,
        temporal_discount: float = 0.25,
        clustering_method: str = "kmeans",
        load_clusters: bool = False
        ) -> np.ndarray:
        """
        Load or generate clusters of embeddings.
        
        Args:
            - num_clusters (int): Number of clusters to generate
            - temporal_discount (float): Discount given to temporal embeddings in kgmeans
            - clustering_method (str): Clustering method to use
            
        Returns:
            - np.ndarray: Clusters of embeddings
        """
        clusters_dir = os.path.join(self.dataset_dir, "clusters", self.filename)
        cluster_filename = f"{num_clusters}_clusters.pkl"
        
        self.num_clusters = num_clusters
        
        if load_clusters:
            self.dataset["clusters"] = utils.load_data(clusters_dir, cluster_filename)
        else:
            self.dataset["clusters"] = self._generate_clusters(
                num_clusters, temporal_discount, clustering_method
                )
            utils.save_data(self.dataset["clusters"], clusters_dir, cluster_filename)
        
        return self.dataset["clusters"]

    def _generate_clusters(
        self,
        num_clusters: int,
        temporal_discount: float = 0.25,
        clustering_method: str = "kmeans"
        ) -> np.ndarray:
        """
        Generate clusters from given fields.
        
        Args:
            - num_clusters (int): Number of clusters to generate
            cluster_by (List[str]): Fields to cluster by from the XRL data
            - temporal_discount (float): Discount given to temporal embeddings in kgmeans
            - clustering_method (str): Clustering method to use
            
        Returns:
            - np.ndarray: Clusters of embeddings
        """
        
        embeddings = self.dataset[self.activation_key]
        if clustering_method == 'kmeans':
            clusters = KMeans(n_clusters=num_clusters,
                              n_init='auto').fit(embeddings).labels_
        elif clustering_method == 'kgmeans':
            # TODO clean up
            dummy_final_y = 2*np.amax(embeddings[:,0]) - np.amin(embeddings[:,0])
            dummy_final_x = 2*np.amax(embeddings[:,1]) - np.amin(embeddings[:,1])
            # Augment each state's embedding with the embedding
            # of the next state in the trajectory
            augmented_embeddings = np.zeros((embeddings.shape[0], 4), dtype=float)
            augmented_embeddings[:,:2] = embeddings
            augmented_embeddings[:-1,2:] = temporal_discount * embeddings[1:,:]
            for final_state in self.dataset['done_indices']:
                augmented_embeddings[final_state,2] = temporal_discount * dummy_final_y
                augmented_embeddings[final_state,3] = temporal_discount * dummy_final_x
            # Run K-Means clustering on the augmented embeddings
            clusters = KMeans(n_clusters=num_clusters, 
                              n_init='auto').fit(augmented_embeddings).labels_
        
        return np.array(clusters)
    
    def embeddings_data(self) -> GraphData:
        """
        Generate data necessary for creating embedding graphs.
        """
        embeddings = self.dataset["embeddings"]
        
        x = embeddings[:,0]
        y = embeddings[:,1]
        colors = ['#5A5A5A'] * len(self.dataset["embeddings"])
        act_key = " ".join(self.activation_key.split("_")).title()
        title = act_key + " Embeddings"
        
        embed_data = GraphData(
            x=x,
            y=y,
            title=title,
            colors=colors,
            showall=False
        )
        
        return embed_data
    
    def cluster_data(self) -> GraphData:
        """
        Generate data necessary for creating cluster graphs.
        """
        embeddings = self.dataset["embeddings"]
        
        x = embeddings[:,0]
        y = embeddings[:,1]
        colors = [utils.CLUSTER_COLORS[i] for i in self.dataset["clusters"]]
        title = f"{self.num_clusters} Clusters"
        
        handles = [Patch(color=utils.CLUSTER_COLORS[i], label=str(i))
                   for i in range(self.num_clusters)]
        labels = [f"Cluster {i}" for i in range(self.num_clusters)]
        leg_title = "Cluster Groups"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        cluster_data = GraphData(
            x=x,
            y=y,
            title=title,
            colors=colors,
            legend=legend,
            showall=False
        )
        
        return cluster_data
    
    def decision_boundary_data(self) -> GraphData:
        """
        Generate data necessary for creating decision boundary graphs.
        """
        embeddings = self.dataset["embeddings"]
        
        x = embeddings[:,0]
        y = embeddings[:,1]
        colors = [utils.CLUSTER_COLORS[i] for i in self.dataset["actions"]]
        title = "Decision Boundaries for Taken Actions"
        
        num_actions = len(np.unique(self.dataset['actions']))
        handles = [Patch(color=utils.CLUSTER_COLORS[i], label=str(i))
                   for i in range(num_actions)]
        labels = [f"{i}" for i in range(num_actions)]
        leg_title = "Action Values"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        decision_boundary_data = GraphData(
            x=x,
            y=y,
            title=title,
            colors=colors,
            legend=legend,
            showall=False
        )
        
        return decision_boundary_data
        
    def episode_prog_data(self) -> GraphData:
        """
        Generate data necessary for creating episode progression graphs.
        """
        embeddings = self.dataset["embeddings"]
        
        x = embeddings[:,0]
        y = embeddings[:,1]
        colors = self.dataset['steps']
        title = "Episode Progression"
        
        episode_prog_data = GraphData(
            x=x,
            y=y,
            title=title,
            colors=colors,
            cmap="viridis",
            showall=False
        )
        
        return episode_prog_data
    
    def confidence_data(self) -> GraphData:
        """
        Generate data necessary for creating episode progression graphs.
        """
        embeddings = self.dataset["embeddings"]
        
        x = embeddings[:,0]
        y = embeddings[:,1]
        colors = np.amax(self.dataset["dist_probs"], axis=1)
        title = "Policy Confidence in Greedy Action"
        
        conf_data = GraphData(
            x=x,
            y=y,
            title=title,
            colors=colors,
            cmap="RdYlGn",
            showall=False
        )
        
        return conf_data
    
    def initial_terminal_state_data(self) -> GraphData:
        """
        Generate data necessary for creating initial/terminal state graphs.
        """
        embeddings = self.dataset["embeddings"]
        
        x = embeddings[:,0]
        y = embeddings[:,1]
        colors = [utils.CLUSTER_COLORS[0] if i else '#F5F5F5' for i in self.dataset["dones"]]
        for i in self.dataset["start_indices"]:
            colors[i] = utils.CLUSTER_COLORS[1] 
        title = "Initial and Terminal States"
        
        handles = [Patch(color=utils.CLUSTER_COLORS[1]),
                   Patch(color=utils.CLUSTER_COLORS[0])]
        labels = ["Initial", "Terminal"]
        leg_title = "State Type"
        
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        state_data = GraphData(
            x=x,
            y=y,
            title=title,
            colors=colors,
            legend=legend,
            showall=False
        )
    
        return state_data
    
    def graph_individual_data(
        self,
        data: GraphData,
        filename: str
        ):
        """Graph given GraphData to a single plot and save a PNG to the given file.

        Args:
            data (GraphData): Data necessary to graph and individual plot.
            filename (str): Name for the PNG file.
        """
        _ = plt.scatter(
            data.x, 
            data.y, 
            c=data.colors, 
            cmap=data.cmap, 
            s=1)
        
        if not data.showall:
            plt.axis('off')
        else:
            plt.xticks(data.x)
            plt.xlabel(data.xlabel)
            plt.ylabel(data.ylabel)
            
        plt.title(data.title)
        
        if data.legend is not None:
            data.legend.update({"bbox_to_anchor": (1.05, 1.0), "loc": 'upper left'})
            plt.legend(**data.legend)
        
        if data.cmap is not None:
            plt.colorbar()
        
        if data.error_bars is not None:
            plt.errorbar(data.x, data.y, yerr=data.error_bars, fmt="o", capsize=5)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.trial_path, "individual_graphs", filename)
        logging.info(f"Saving individual graph png to {save_path}...")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def graph_analytics(self,
                        data: List[GraphData],
                        horizontal: bool = True) -> None:
        """Graph all given GraphData to a single figure with multiple subplots.

        Args:
            data (List[GraphData]): Collection of GraphData create subplots for.
            horizontal (bool, optional): Place more subplots horizontally to create
                a more horizontal figure than vertical. Defaults to True.
        """
        
        utils.graph_subplots(
            "XAI Data Analysis", 
            data, 
            self.trial_path, 
            "xai_data_analytics.png",
            horizontal)
    
    def get_SAMDP(
        self,
        load_samdp: bool = False
        ) -> nx.Graph:
        """Create or load an SAMDP from the current dataset.

        Args:
            load_samdp (bool, optional): To load a previously generated SAMDP graph. 
                Defaults to False.

        Returns:
            nx.Graph: NetworkX Graph representation of the SAMDP.
        """
        
        samdp_dir = os.path.join(self.dataset_dir, "samdp")
        samdp_path = os.path.join(samdp_dir, "samdp.graphml")
        
        if load_samdp:
            self.graph = nx.read_graphml(samdp_path)
        else:
            self.graph = self._generate_SAMDP()
            if not os.path.exists(samdp_dir):
                os.makedirs(samdp_dir)
            nx.write_graphml(self.graph, samdp_path)
            
        return self.graph
    
    def _generate_SAMDP(self) -> nx.Graph:
        """Create an SAMDP from the current dataset.

        Raises:
            ValueError: Returned if there are no clusters created to serve as nodes.

        Returns:
            nx.Graph: NetworkX Graph representation of the SAMDP.
        """
        
        try:
            cluster_data = self.dataset['clusters']
            actions = self.dataset['actions']
        except:
            raise ValueError("There are no clusters to create the SAMDP of.")
        
        num_actions = len(np.unique(actions))
        samdp_counts = np.zeros([self.num_clusters, num_actions, self.num_clusters])
        
        for i in range(len(cluster_data) - 1):
            done = self.dataset['dones'][i]
            
            if not done:
                cur_cluster = cluster_data[i]
                action = actions[i]
                next_cluster = cluster_data[i+1]
            
                samdp_counts[cur_cluster, action, next_cluster] += 1
        
        np.set_printoptions(suppress=True)
        np.seterr(divide='ignore', invalid='ignore')
        
        totals_movements = np.sum(samdp_counts, axis=-1)
        totals_movements = np.expand_dims(totals_movements, axis=-1)
        totals_movements = np.repeat(totals_movements, self.num_clusters, axis=-1)
        
        samdp = samdp_counts / totals_movements
        self.samdp = np.nan_to_num(samdp, nan=0)

        self._create_SAMDP_txt(samdp_counts)
        G = self._create_SAMDP_graph()
        
        return G

    def _create_SAMDP_txt(self, samdp_counts: np.ndarray) -> None:
        """Create a txt table of the SAMDP.

        Args:
            samdp_counts (np.ndarray): The number of times an agent moved from one
                cluster to another along with the action taken to get to the latter.
        """
        num_actions = samdp_counts.shape[1]
        samdp_data = ["SAMDP"]
        for from_cluster_id in range(self.num_clusters):
            table = PrettyTable()
            table.title = f"Cluster {from_cluster_id}"
            
            headers = [f"Cluster {i}" for i in range(self.num_clusters)]
            table.field_names = ["Action Value"] + headers
            for action in range(num_actions):
                row = [f'Action {action}']
                
                for to_cluster_id in range(self.num_clusters):
                    value = samdp_counts[from_cluster_id, action, to_cluster_id]
                    percent = self.samdp[from_cluster_id, action, to_cluster_id]
                    row.append(f"{value} | {round(percent*100, 2)}%")
                table.add_row(row)
            
            samdp_data.append(str(table))
        
        samdp_data = "\n".join(samdp_data)
        
        with open(os.path.join(self.trial_path, 'SAMDP', 'samdp.txt'), 'w') as f:
            f.write(samdp_data)
    
    def _create_SAMDP_graph(self) -> nx.Graph:
        """Create a graph of this dataset's SAMDP using NetworkX.
        
        Each node represents a cluster from self.dataset['clusters'] and the edges
        represent the paths the agent takes in the dataset between clusters. An edge is
        added for each action taken that brings the agent from one cluster to another.
        For each action from a cluster, only the edge with the highest probability is 
        shown, meaning there are other clusters that action can move the agent to but
        only the highest probability edge is shown.

        Returns:
            nx.Graph: NetworkX Graph representation of the SAMDP
        """
        
        num_actions = self.samdp.shape[1]
        
        _ = plt.figure(figsize=(40,20))
        plt.title('SAMDP')
        G = nx.MultiDiGraph()
        
        G.add_nodes_from([f"Cluster {i}" for i in range(self.num_clusters)])
        
        edges = []
        edge_colors = []
        for from_cluster_id in range(self.num_clusters):
            from_cluster = f"Cluster {from_cluster_id}"
            for action_id in range(num_actions):
                best_edge = {'prob': 0}
                for to_cluster_id in range(self.num_clusters):
                    to_cluster = f"Cluster {to_cluster_id}"
                    
                    prob = self.samdp[from_cluster_id, action_id, to_cluster_id]
                    if not prob == 0 and not from_cluster_id == to_cluster_id:
                        edge = (from_cluster, to_cluster)
                        prob_percent = round(prob * 100, 2)
                        
                        if prob_percent > best_edge["prob"]:
                            best_edge["edge"] = edge
                            best_edge["prob"] = prob_percent.item()
                
                if not best_edge["prob"] == 0:
                    edges.append(best_edge['edge'])   
                    G.add_edge(best_edge['edge'][0], 
                               best_edge['edge'][1], 
                               weight=best_edge['prob'],
                               action=action_id)

        sources = []
        for node in G.nodes():
            if G.in_degree(node) == 0:
                sources.append(node)
        
        pos = {}
        bfs_layers = list(nx.bfs_layers(G, sources))
        for i, layer_list in enumerate(bfs_layers):
            for j, node in enumerate(layer_list):
                pos[node] = (i, j)
        
        same_layer_perms = []
        for layer in bfs_layers:
            same_layer_perms += list(itertools.permutations(layer, 2))
        
        edge_arcs = []
        edges = G.edges()
        for edge in G.edges:
            e = (edge[0], edge[1])
            if e in same_layer_perms or (edge[1], edge[0]) in edges:
                edge_arcs.append(0.1*(edge[2]+1))
            else:
                edge_arcs.append(0)
            
        nx.draw_networkx_nodes(G, 
                               pos,
                               node_size=4000,
                               node_color=utils.CLUSTER_COLORS[0:self.num_clusters])
        
        nx.draw_networkx_labels(G, pos)
        
        samdp_edges_data = list(G.edges(data=True))
        samdp_edges = list(G.edges)
        
        for i, edge in enumerate(G.edges):
            edge_index = samdp_edges.index(edge)
            edge_data = samdp_edges_data[edge_index]
            action_id = edge_data[2]["action"]
            nx.draw_networkx_edges(G, 
                                   pos, 
                                   edgelist=[edge], 
                                   connectionstyle=f"arc3,rad={edge_arcs[i]}",
                                   edge_color=utils.CLUSTER_COLORS[action_id],
                                   node_size=4000, 
                                   arrowsize=25)
        
        plt.tight_layout()
        save_path = os.path.join(self.trial_path, "SAMDP", "SAMDP.png")
        logging.info(f"Saving SAMDP graph png to {save_path}...")
        plt.savefig(save_path, format="PNG")
        plt.close()
        
        return G
    
    def calculate_path_probs(self, paths: List[Tuple[str, str, int]]) -> Dict[int, float]:
        """Calculate the probability of each path being taken.

        Args:
            paths (List[Tuple[str, str, int]]): All simple paths from one cluster to 
                another.

        Returns:
            Dict[int, float]: Dictionary with actions as keys and highest probability
                to reach target from current node.
        """
        samdp_edges = list(self.graph.edges)
        samdp_edges_data = list(self.graph.edges(data=True))
        
        probs = {}
        for path in paths:
            prob = 1
            edge_index = samdp_edges.index(path[0])
            edge_data = samdp_edges_data[edge_index]
            action = edge_data[2]['action']
            
            for edge in path:
                edge_index = samdp_edges.index(edge)
                edge_data = samdp_edges_data[edge_index]
                edge_prob = edge_data[2]['weight'] / 100
                prob = prob * edge_prob
            
            if action in probs:
                if prob > probs[action]:
                    probs[action] = prob
            else:
                probs[action] = prob
            logging.info(f'\tPath via {action}: {round(prob * 100,2)}%')
        
        return probs
    
    def analyze_clusters(self):
        
        cluster_conf = [[] for _ in range(self.num_clusters)]
        
        for e, i in enumerate(self.dataset["clusters"]):
            conf = np.amax(self.dataset['dist_probs'][e]).astype(np.float64)
            cluster_conf[i].append(conf)
            
        means = []
        stdevs = []
        
        for i in range(self.num_clusters):
            means.append(statistics.mean(cluster_conf[i]))
            stdevs.append(statistics.stdev(cluster_conf[i]))

        
        title = "Cluster Confidence Analysis"
        
        cluster_conf_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            error_bars=stdevs,
            xlabel='Cluster ID',
            ylabel='Mean Highest Action Confidence',
            showall=True
        )
        
        return cluster_conf_data
    
    def find_paths(self, from_cluster_id: int, to_cluster_id: int):
        """Find simple paths between two clusters within SAMDP.

        Args:
            from_cluster_id (int): ID of cluster to start from.
            to_cluster_id (int): ID of cluster to end at.
        """
        from_cluster = f'Cluster {from_cluster_id}'
        to_cluster = f'Cluster {to_cluster_id}'
        logging.info(f'Finding paths from {from_cluster} to {to_cluster}...')
        paths = list(nx.all_simple_edge_paths(self.graph, from_cluster, to_cluster))
        
        probs = self.calculate_path_probs(paths)
        
        logging.info(f"Highest probability of getting from {from_cluster} to {to_cluster}:")
        for action in probs:
            logging.info(f"\tvia Action {action}: {round(probs[action] * 100, 2)}%")
        best_action = max(probs, key=probs.get)
        logging.info(f'\tBest Option: Action {best_action} with {round(probs[best_action] * 100, 2)}%')
        
        _ = plt.figure(figsize=(15,15))
        plt.title(f'Paths from {from_cluster} to {to_cluster}')
        G = nx.MultiDiGraph()
        
        samdp_edges = list(self.graph.edges)
        samdp_edges_data = list(self.graph.edges(data=True))
        
        edges = []
        edge_colors = []
        node_colors = []
        for path in paths:
            for edge in path:
                if edge in edges:
                    continue
                
                if edge[0] not in G.nodes:
                    G.add_node(edge[0])
                    node_id = int(edge[0].split(' ')[-1])
                    node_colors.append(utils.CLUSTER_COLORS[node_id])
                    
                    
                if edge[1] not in G.nodes:
                    G.add_node(edge[1])
                    node_id = int(edge[1].split(' ')[-1])
                    node_colors.append(utils.CLUSTER_COLORS[node_id])
                    
                edge_index = samdp_edges.index(edge)
                edge_data = samdp_edges_data[edge_index]
                
                edges.append(edge)
                G.add_edge(
                    edge[0], 
                    edge[1], 
                    weight=edge_data[2]['weight'], 
                    action=edge_data[2]['action']
                    )
        
        pos = nx.shell_layout(G)
        nx.draw_networkx_nodes(G, 
                               pos,
                               node_size=4000,
                               node_color=node_colors)
        
        nx.draw_networkx_labels(G, pos)
        
        for edge in G.edges:
            edge_index = samdp_edges.index(edge)
            edge_data = samdp_edges_data[edge_index]
            action_id = edge_data[2]["action"]
            nx.draw_networkx_edges(G, 
                                   pos, 
                                   edgelist=[edge], 
                                   connectionstyle=f"arc3,rad={0.1*edge[2]}",
                                   edge_color=utils.CLUSTER_COLORS[action_id],
                                   node_size=4000, 
                                   arrowsize=25)
        
        plt.tight_layout()
        save_path = os.path.join(self.trial_path, "SAMDP", f"{from_cluster}-{to_cluster}.png")
        logging.info(f"Saving paths from {from_cluster} to {to_cluster} graph png to {save_path}...")
        plt.savefig(save_path, format="PNG")
        plt.close()        
                