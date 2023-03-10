from typing import Any, Dict
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import os
import pickle
import logging

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import perfect_timing.utils.data_analysis_utils as utils

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
        load_embeddings: bool = False
        ) -> np.ndarray:
        """
        Load pre-made embeddings or generate new embeddings from an activation key.
        
        Args:
            - activation_key (str): Data to perform dimensionality reduction on
            - num_components (int): Number of components to reduce to
            - perplexity (int): Value for balancing local vs global aspects
            - load_embeddings (bool): Wheter or not to load a premade embedding
            
        Returns:
            - np.ndarray: Embeddings of values from the activation key
        """
        
        embeddings_dir = os.path.join(self.dataset_dir, "embeddings")
        embed_filename = f"{self.filename}-{activation_key}-{perplexity}_embeddings.pkl"
        
        self.perplexity = perplexity
        self.activation_key = activation_key
        
        if load_embeddings:
            self.dataset['embeddings'] = utils.load_data(embeddings_dir, embed_filename)
        else:
            
            self.dataset['embeddings'] = self._generate_embeddings(
                activation_key, num_components, perplexity,
            )
            utils.save_data(self.dataset['embeddings'], embeddings_dir, embed_filename)
            
        return self.dataset['embeddings']
    
    def _generate_embeddings(
        self,
        activation_key: str, 
        num_components: int,
        perplexity: int
        ) -> np.ndarray:
        """
        Generate new embeddings from an activation key.
        
        Args:
            - activation_key (str): Data to perform dimensionality reduction on
            - num_components (int): Number of components to reduce to
            - perplexity (int): Value for balancing local vs global aspects
            - load_embeddings (bool): Wheter or not to load a premade embedding
        
        Returns:
            - np.ndarray: Embeddings of values from the activation key
        """
        
        embedder = TSNE(
            n_components=num_components, 
            perplexity=perplexity, 
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
        temporal_discount: float,
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
        clusters_dir = os.path.join(self.dataset_dir, "clusters")
        cluster_filename = f"{self.filename}-{self.activation_key} \
            -{num_clusters}_clusters.pkl"
        
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
        temporal_discount: float,
        clustering_method: str = "kmeans"
        ) -> np.ndarray:
        """
        Generate clusters of embeddings.
        
        Args:
            - num_clusters (int): Number of clusters to generate
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
    
    def graph_clusters(self) -> None:
        """
        Graph and save an image of the embedding clusters.
        """
        
        colors = [utils.CLUSTER_COLORS[i] for i in self.dataset["clusters"]]
        embeddings = self.dataset["embeddings"]
        
        _ = plt.scatter(embeddings[:,0], embeddings[:,1], c=colors, s=1)
        plt.axis('off')
        title = f"{self.filename} {self.activation_key} Embeddings "\
            f"with {self.num_clusters} Groups and {self.perplexity} Perplexity"
        plt.title(title)
        
        handles = [Patch(color=utils.CLUSTER_COLORS[i], label=str(i))
                   for i in range(self.num_clusters)]
        labels = [f"Group {i}" for i in range(self.num_clusters)]
        plot_title = "Cluster Groups"
        plt.legend(handles=handles, labels=labels, loc="lower right", title=plot_title)
        
        save_dir = './outputs/cluster_graphs/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_filename = f'{self.filename}-{self.activation_key}- \
        {self.num_clusters}-{self.perplexity}'
        plt.savefig(os.path.join(save_dir, plot_filename))
    
    def graph_embeddings(self) -> None:
        """
        Graph and save an image of the embeddings.
        """
        embeddings = self.dataset["embeddings"]
        
        _ = plt.scatter(embeddings[:,0], embeddings[:,1], s=1)
        plt.axis('off')
        title = f"{self.filename} {self.activation_key} Embeddings"\
            f" with {self.perplexity} Perplexity"
        plt.title(title)
        
        save_dir = './outputs/embedding_graphs/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_filename = f'{self.filename}-{self.activation_key}-{self.perplexity}'
        plt.savefig(os.path.join(save_dir, plot_filename))
        
    def graph_action_embeddings(self) -> None:
        """
        Graph and save an image of the embeddings grouped by action.
        """
        embeddings = self.dataset["embeddings"]
        
        colors = [utils.CLUSTER_COLORS[i] for i in self.dataset["actions"]]
        possible_actions = np.unique(self.dataset["actions"])
        
        _ = plt.scatter(embeddings[:,0], embeddings[:,1], c=colors, s=1)
        plt.axis('off')
        title = f"{self.filename} {self.activation_key} Embeddings"\
            f" with {self.perplexity} Perplexity"
        plt.title(title)
        
        handles = [Patch(color=utils.CLUSTER_COLORS[i], label=str(i))
                   for i in possible_actions]
        labels = [f"Action Value {i}" for i in possible_actions]
        plot_title = "Actions"
        plt.legend(handles=handles, labels=labels, loc="lower right", title=plot_title)
        
        save_dir = './outputs/action_graphs/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_filename = f'{self.filename}-{self.activation_key}-{self.perplexity}_actions'
        plt.savefig(os.path.join(save_dir, plot_filename))