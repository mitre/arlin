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
    dataset: XRLDataset,
    num_clusters: int,
    cluster_key: Optional[str] = None,
    embeddings: Optional[np.ndarray] = None,
    clustering_method: str = 'kmeans'
) -> np.ndarray:
    
    if cluster_key is None and embeddings is None:
        raise ValueError('At least one of cluster_key and embeddings must be set.')
    
    if cluster_key is not None and embeddings is not None:
        raise ValueError('Only one of cluster_key and embeddings can be set at a time.')
    
    if cluster_key is not None:
        try:
            cluster_on = getattr(dataset, cluster_key)
        except:
            raise ValueError(f"Cluster key {cluster_key} is not in dataset.")
    else:
        cluster_on = embeddings
    
    if clustering_method == 'kmeans':
        clusters = KMeans(n_clusters=num_clusters,
                            n_init='auto').fit(cluster_on).labels_
    else:
        raise NotImplementedError(f'{clustering_method} is not currently supported.')
    
    return np.array(clusters)