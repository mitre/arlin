import numpy as np
import logging
from arlin.data_analysis.xrl_dataset import XRLDataset
from sklearn.manifold import TSNE
from sklearn.cluster import (
    KMeans, 
    AgglomerativeClustering, 
    SpectralClustering, 
    OPTICS, 
    MeanShift
)
import time

np.random.seed(1234)

def generate_embeddings(
    dataset: XRLDataset,
    activation_key: str,
    perplexity: int,
    n_train_iter: int,
    output_dim: int = 2,
    seed: int = 12345
    ) -> np.ndarray:
    
    logging.info(f"Generating embeddings from dataset.{activation_key}.")
    
    start = time.time()
    
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
    
    end = time.time()
    
    logging.info(f"Successfully generated embeddings in {end - start} seconds.")
    
    return np.array(embeddings)

def generate_clusters(
    cluster_on: np.ndarray,
    num_clusters: int,
    clustering_method: str = 'kmeans'
) -> np.ndarray:
    
    logging.info(f"Generating clusters using the {clustering_method} method.")
    
    start = time.time()
    
    clustering_method = clustering_method.lower()
    if clustering_method == 'kmeans':
        clusters = KMeans(n_clusters=num_clusters,
                          n_init='auto').fit(cluster_on)
        
    elif clustering_method == 'hac':
        clusters = AgglomerativeClustering(n_clusters=num_clusters, 
                                           metric='euclidean', 
                                           linkage='ward').fit(cluster_on)
        
    elif clustering_method == 'spectral':
        clusters = SpectralClustering(n_clusters=num_clusters, 
                                      assign_labels='cluster_qr', 
                                      random_state=0).fit(cluster_on)
    
    elif clustering_method == 'optics':
        clusters = OPTICS(min_samples=500).fit(cluster_on)
        
    elif clustering_method == 'meanshift':
        clusters = MeanShift().fit(cluster_on)
    
    else:
        raise NotImplementedError(f'{clustering_method} is not currently supported.')
    
    end = time.time()
    logging.info(f"Successfully generated clusters in {end - start} seconds.")
    
    return clusters.labels_