from perfect_timing.graph_creation.graph_data import GraphData

from typing import Any, Dict
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import perfect_timing.utils.graph_creator_utils as utils

class GraphCreator():
    def __init__(
        self, 
        load_embeddings: bool,
        load_clusters: bool,
        num_components: int,
        perplexity: int,
        tsne_init: str,
        tsne_seed: int,
        num_clusters: int,
        GRAPH_DATA: Dict[str, Any]
        ):
        self.graph_data = GraphData(**GRAPH_DATA)
        
        dataset_dir = os.path.dirname(self.graph_data.dataset_path)
        filename = os.path.basename(self.graph_data.dataset_path).split('.')[0]
        embeddings_dir = os.path.join(dataset_dir, "embeddings")
        embed_filename = f"{filename}-{self.graph_data.activation_key}_embeddings.pkl"
        
        if load_embeddings:
            self.embeddings = utils.load_data(embeddings_dir, embed_filename)
        else:
            self.embeddings = self._get_embeddings(
                num_components, 
                perplexity, 
                tsne_init, 
                tsne_seed)
            utils.save_data(self.embeddings, embeddings_dir, embed_filename)

        clusters_dir = os.path.join(dataset_dir, "clusters")
        cluster_filename = f"{filename}-{num_clusters}_clusters.pkl"
        if load_clusters:
            self.cluster_groups = utils.load_data(clusters_dir, cluster_filename)
        else:
            self.cluster_groups = self._get_clusters(num_clusters)
            utils.save_data(self.cluster_groups, clusters_dir, cluster_filename)

        self.graph_clusters(filename, num_clusters)
    
    
    def graph_clusters(self, data_filename: str, num_clusters: int) -> None:
        
        colors = [utils.CLUSTER_COLORS[i] for i in self.cluster_groups]
        _ = plt.scatter(self.embeddings[:,0], self.embeddings[:,1], c=colors, s=1)
        plt.axis('off')
        plt.title(f"{data_filename} with {num_clusters} Groups")
        
        unique_groups = np.unique(self.cluster_groups)
        
        handles = [Patch(color=utils.CLUSTER_COLORS[i], label=str(i)) for i in unique_groups]
        
        plt.legend(handles=handles, labels=[f"Group {i}" for i in unique_groups], loc="lower right", title="Cluster Groups")
        
        save_dir = './outputs/cluster_graphs/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        plt.savefig(os.path.join(save_dir, 'test.png'))
    
    
    def _get_clusters(self, num_clusters: int) -> np.ndarray:
        # cluster_groups = KMeans(
        #     n_clusters=num_clusters, 
        #     n_init='auto').fit(self.embeddings).labels_
        
        temporal_discount = 0.25
        # Create a dummy "ultra-final" state that will immediately follow each actual final state
        dummy_final_y = 2*np.amax(self.embeddings[:,0]) - np.amin(self.embeddings[:,0])
        dummy_final_x = 2*np.amax(self.embeddings[:,1]) - np.amin(self.embeddings[:,1])
        # Augment each state's embedding with the embedding of the next state in the trajectory
        augmented_embeddings = np.zeros((self.embeddings.shape[0], 4), dtype=float)
        augmented_embeddings[:,:2] = self.embeddings
        augmented_embeddings[:-1,2:] = temporal_discount * self.embeddings[1:,:]
        for final_state in self.graph_data.done_indices:
            augmented_embeddings[final_state,2] = temporal_discount * dummy_final_y
            augmented_embeddings[final_state,3] = temporal_discount * dummy_final_x
        # Run K-Means clustering on the augmented embeddings
        cluster_groups = KMeans(n_clusters=num_clusters).fit(augmented_embeddings).labels_
        
        return np.array(cluster_groups)
        
    def _get_embeddings(
        self,
        num_components: int,
        perplexity: int,
        tsne_init: str, 
        tsne_seed: int
        ) -> np.ndarray:
        embedder = TSNE(
            n_components=num_components, 
            init=tsne_init, 
            perplexity=perplexity, 
            verbose=1, 
            random_state=tsne_seed)
        
        unique_activations = self.graph_data.activations[self.graph_data.unique_indices]
        embeddings = embedder.fit_transform(unique_activations)
        embeddings = np.array([embeddings[index] for index in self.graph_data.state_mapping])
        return embeddings
