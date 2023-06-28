from typing import Dict, Optional
import os
import numpy as np
from arlin.data_analysis.xrl_dataset import XRLDataset
from arlin.data_analysis.samdp import SAMDP
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import networkx as nx

np.random.seed(1234)

def generate_embeddings(
    dataset: XRLDataset,
    activation_key: str,
    perplexity: int,
    n_train_iter: int,
    output_dim: int = 2,
    seed: int = 12345
    ) -> np.ndarray:
    
    embedder = TSNE(
        n_components=output_dim, 
        perplexity=perplexity,
        n_iter=n_train_iter,
        verbose=1, 
        random_state=seed)

    activations = getattr(dataset, activation_key)
    unique_activations = activations[dataset.unique_state_indices]
    
    embeddings = embedder.fit_transform(unique_activations)
    embeddings = [embeddings[index] for index in dataset.state_mapping]
    
    return np.array(embeddings)

def generate_clusters(
    cluster_on: np.ndarray,
    num_clusters: int,
    clustering_method: str = 'kmeans'
) -> np.ndarray:
    clustering_method = clustering_method.lower()
    if clustering_method == 'kmeans':
        clusters = KMeans(n_clusters=num_clusters,
                            n_init='auto').fit(cluster_on).labels_
    else:
        raise NotImplementedError(f'{clustering_method} is not currently supported.')
    
    return clusters