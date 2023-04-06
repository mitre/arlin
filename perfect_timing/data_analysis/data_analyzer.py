from typing import Any, Dict, List
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import os
import pickle
import logging

import matplotlib as mpl
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
        cluster_by: List[str],
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
        cluster_name = "_".join(cluster_by)
        clusters_dir = os.path.join(self.dataset_dir, "clusters",
                                    self.filename, cluster_name)
        cluster_filename = f"{num_clusters}_clusters.pkl"
        
        self.num_clusters = num_clusters
        
        if load_clusters:
            self.dataset["clusters"] = utils.load_data(clusters_dir, cluster_filename)
        else:
            self.dataset["clusters"] = self._generate_clusters(
                num_clusters, cluster_by, temporal_discount, clustering_method
                )
            utils.save_data(self.dataset["clusters"], clusters_dir, cluster_filename)
        
        return self.dataset["clusters"]
    
    def _concatenate_data(self, cluster_by: List[str]):
        
        # only pull unique data
        cluster_by_data = []
        for field in cluster_by:
            unique_field_data = self.dataset[field][self.dataset["unique_indices"]]
            
            if len(unique_field_data.shape) == 1:
                unique_field_data = np.expand_dims(unique_field_data, axis=-1)
            cluster_by_data.append(unique_field_data)
            
        datapoints = np.concatenate(cluster_by_data, axis=-1)
        
        self.dataset["clustered_data"] = datapoints
        
        return datapoints
            

    def _generate_clusters(
        self,
        num_clusters: int,
        cluster_by = List[str],
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
        
        #data = self._concatenate_data(cluster_by)
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
        embeddings = self.dataset["embeddings"]
        colors = [utils.CLUSTER_COLORS[i] for i in self.dataset["clusters"]]
        
        _ = plt.scatter(embeddings[:,0], embeddings[:,1], c=colors, s=1, alpha=0.1)
        plt.axis('off')
        title = f"{self.filename} {self.activation_key} Embeddings "\
            f"with {self.num_clusters} Groups and {self.perplexity} Perplexity"
        plt.title(title)
        
        handles = [Patch(color=utils.CLUSTER_COLORS[i], label=str(i))
                   for i in range(self.num_clusters)]
        labels = [f"{i}" for i in range(self.num_clusters)]
        plot_title = "Cluster Groups"
        plt.legend(handles=handles, labels=labels, loc="lower center", title=plot_title, ncol=self.num_clusters)
        plt.tight_layout()
        
        save_dir = f'./outputs/cluster_graphs/{self.filename}/{self.activation_key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_filename = f"{self.perplexity}_perplexity-{self.n_iter}_iter-"\
            f"{self.num_clusters}_clusters"
        logging.info(f"Saving cluster graph png to {plot_filename}...")
        plt.savefig(os.path.join(save_dir, plot_filename), bbox_inches='tight')
        
    def graph_action_embeddings(self) -> None:
        """
        Graph and save an image of the embeddings grouped by action.
        """
        embeddings = self.dataset["embeddings"]
        action_probs = np.amax(self.dataset["dist_probs"], axis=1)
        values = self.dataset['critic_values']
        
        colors = [utils.CLUSTER_COLORS[i] for i in self.dataset["actions"]]
        possible_actions = np.unique(self.dataset["actions"])
        
        fig, axs = plt.subplots(2,2)
        title = f"{self.filename} {self.activation_key} Embeddings"\
            f" with {self.perplexity} Perplexity"
        fig.suptitle(title)
        
        _ = axs[0,0].scatter(embeddings[:,0], embeddings[:,1], c=colors, s=0.1)
        axs[0,0].axis('off')
        axs[0,0].set_title("Taken Actions")
        axs[0,0].title.set_size(10)
        
        # handles = [Patch(color=utils.CLUSTER_COLORS[i], label=str(i))
        #            for i in possible_actions]
        # labels = [f"{i}" for i in possible_actions]
        # plot_title = "Actions"
        # axs[0,0].legend(handles=handles, labels=labels, loc="lower right", title=plot_title)
        
        #---------
        
        _ = axs[0,1].scatter(embeddings[:,0], embeddings[:,1], cmap="RdYlGn", c=action_probs, s=0.1)
        axs[0,1].axis('off')
        axs[0,1].set_title("Action Confidence Heatmap")
        axs[0,1].title.set_size(10)
        
        #---------
        
        _ = axs[1,0].scatter(embeddings[:,0], embeddings[:,1], cmap="RdYlGn", c=values, s=0.1)
        axs[1,0].axis('off')
        axs[1,0].set_title("Action Importance Heatmap")
        axs[1,0].title.set_size(10)
        
        #---------
        
        colors = [utils.CLUSTER_COLORS[0] if i else '#F5F5F5' for i in self.dataset["dones"]]
        for i in self.dataset["start_indices"]:
            colors[i] = utils.CLUSTER_COLORS[1] 
        
        _ = axs[1,1].scatter(embeddings[:,0], embeddings[:,1], c=colors, s=0.1)
        axs[1,1].axis('off')
        axs[1,1].set_title("Initial and Terminal States")
        axs[1,1].title.set_size(10)
        
        # handles = [Patch(color=utils.CLUSTER_COLORS[0]),
        #            Patch(color=utils.CLUSTER_COLORS[1])]
        # labels = ["Initial", "Terminal"]
        # plot_title = "States"
        # axs[1,1].legend(handles=handles, labels=labels, loc="lower right", title=plot_title)
        
        plt.tight_layout()
        
        save_dir = f'./outputs/action_graphs/{self.filename}/{self.activation_key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_filename = f'{self.perplexity}_perplexity-{self.n_iter}_iter-actions'
        logging.info(f"Saving action graph png to {plot_filename}...")
        plt.savefig(os.path.join(save_dir, plot_filename))
        plt.close()
        
    def graph_latents(self) -> None:
        """
        Graph and save an image of the embeddings grouped by action.
        """
        embeddings = self.dataset["embeddings"]
        
        #colors = [utils.CLUSTER_COLORS[i] for i in self.dataset["actions"]]
        
        from matplotlib import cm
        cmap = cm.get_cmap('viridis')
        
        vals = []
        v = 1
        skip = False
        for i in range(50000):
            if skip:
                skip = False
                continue
            action = self.dataset["actions"][i]
            
            if action == 1:
                vals.append(v)
                v += 1
                vals.append(v)
                v += 1
                skip = True
            else:
                v = 1
                vals.append(0)
                skip = False
            
        vals = np.array(vals[:-1])       
        norms = (vals - 1) / (vals.max() - 1)
        colors = cmap(norms)
        
        for i in range(len(vals)):
            if vals[i] == 0:
                colors[i] = [0.9,0.9,0.9,1]
        
        # p = []
        # step = 0
        # for i in self.dataset["dones"]:
        #     p.append(step)
        #     step +=1
        #     if i:
        #         v = step
        #         step = 0
        #         break
        
        # s = [0] * (50000 - v)   
        # p.extend(s)
        # self.dataset["steps"] = np.array(p)
        
        # d = self.dataset["steps"][0:v]
        # norms = (d - d.min()) / (d.max() - d.min())
        
        # add = [cmap(norms[i]) for i in d]
        
        # colors = [[0.9,0.9,0.9,1] for i in range((50000 - v))]
        
        # add.extend(colors)
        
        fig, axs = plt.subplots(1)
        title = f"{self.filename} {self.activation_key} Embeddings"\
            f" with {self.perplexity} Perplexity"
        fig.suptitle(title)
        
        _ = axs.scatter(embeddings[:,0], embeddings[:,1], cmap='viridis', c=colors, s=0.1)
        axs.axis('off')
        axs.set_title("Taken Actions")
        axs.title.set_size(10)
        
        plt.tight_layout()
        
        save_dir = f'./outputs/test_graphs/{self.filename}/{self.activation_key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plot_filename = f'{self.perplexity}_perplexity-{self.n_iter}_iter-actions-lefts'
        logging.info(f"Saving action graph png to {plot_filename}...")
        plt.savefig(os.path.join(save_dir, plot_filename))
        plt.close()