from typing import Any, Dict, List
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import os
import pickle
import logging
import pathlib
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import arlin.utils.data_analysis_utils as utils
from arlin.data_analysis.graph_data import GraphData

class DataAnalyzer():
    """
    #### Description
    Class for analyzing, embedding, clustering, and visualizing RL dataset information.
    """
    
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
        logging.info(f"Found {unique_states.shape[0]} distinct states!")
        
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
        
        embeddings = self.dataset['embeddings']
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
            showaxis=False
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
            showaxis=False
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
            showaxis=False
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
            showaxis=False
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
        
        handles = [Patch(color=utils.CLUSTER_COLORS[0]),
                   Patch(color=utils.CLUSTER_COLORS[1])]
        labels = ["Initial", "Terminal"]
        leg_title = "State Type"
        
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        state_data = GraphData(
            x=x,
            y=y,
            title=title,
            colors=colors,
            legend=legend,
            showaxis=False
        )
    
        return state_data
    
    def graph_individual_data(
        self,
        data: GraphData,
        filename: str
        ):
        _ = plt.scatter(
            data.x, 
            data.y, 
            c=data.colors, 
            cmap=data.cmap, 
            s=1)
        
        if not data.showaxis:
            plt.axis('off')
        plt.title(data.title)
        
        if data.legend is not None:
            data.legend.update({"bbox_to_anchor": (1.05, 1.0), "loc": 'upper left'})
            plt.legend(**data.legend)
        
        if data.cmap is not None:
            plt.colorbar()
        
        plt.tight_layout()
        
        save_path = os.path.join(self.trial_path, "individual_graphs", filename)
        logging.info(f"Saving individual graph png to {save_path}...")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def graph_analytics(self) -> None:
        
        db_data = self.decision_boundary_data()
        conf_data = self.confidence_data()
        ep_prog_data = self.episode_prog_data()
        initial_terminal_states = self.initial_terminal_state_data()
        
        data = [db_data, conf_data, ep_prog_data, initial_terminal_states]
        
        utils.graph_subplots(
            "XAI Data Analysis", 
            data, 
            self.trial_path, 
            "xai_data_analytics.png")