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
from tslearn.clustering import TimeSeriesKMeans
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
    
    logging.info(f"\tSuccessfully generated embeddings in {end - start} seconds.")
    
    return np.array(embeddings)

def _get_cluster_ons(dataset: XRLDataset, embeddings: np.ndarray):
    cluster_on_start = dataset.critic_values[dataset.start_indices].reshape(-1, 1)
    cluster_on_done = dataset.total_rewards[dataset.done_indices].reshape(-1, 1)
    
    mask = np.ones([len(dataset.dones)], dtype=bool)
    mask[dataset.start_indices] = False
    mask[dataset.done_indices] = False
    
    embeddings = embeddings[mask]
    actions = np.expand_dims(dataset.actions[mask], axis=-1)
    values = np.expand_dims(dataset.critic_values[mask], axis=-1)
    steps = np.expand_dims(dataset.steps[mask], axis=-1)
    rewards = np.expand_dims(dataset.rewards[mask], axis=-1)
    total_rewards = np.expand_dims(dataset.total_rewards[mask], axis=-1)
    confidences = np.expand_dims(np.amax(dataset.dist_probs, axis=1)[mask], axis=-1)
    
    cluster_on = np.concatenate([embeddings,
                                 actions,
                                 values,
                                 steps,
                                 rewards,
                                 total_rewards,
                                 confidences], axis=-1)
    
    return cluster_on, mask, cluster_on_start, cluster_on_done

def generate_clusters(
    dataset: XRLDataset,
    embeddings: np.ndarray,
    num_clusters: int
) -> np.ndarray:
    
    logging.info(f"Generating {num_clusters} clusters.")
    
    start = time.time()
    
    (cluster_on, 
     cluster_on_mask, 
     cluster_on_start, 
     cluster_on_done) = _get_cluster_ons(dataset, embeddings)
    
    start_clusters = MeanShift().fit(cluster_on_start)
    done_clusters = MeanShift().fit(cluster_on_done)
    mid_clusters = KMeans(n_clusters=num_clusters, n_init='auto').fit(cluster_on)
    
    start_clusters = start_clusters.labels_
    done_clusters = done_clusters.labels_
    mid_clusters = mid_clusters.labels_
    
    n_clusters = len(set(mid_clusters))
    n_start_clusters = len(set(start_clusters))
    
    start_clusters = np.array([x+n_clusters for x in start_clusters], dtype=int)
    done_clusters = np.array([x+n_start_clusters+n_clusters for x in done_clusters], dtype=int)
    
    clusters = np.empty([len(dataset.dones)], dtype=int)
    clusters[cluster_on_mask] = mid_clusters
    clusters[dataset.start_indices] = start_clusters
    clusters[dataset.done_indices] = done_clusters
    
    end = time.time()
    logging.info(f"\tSuccessfully generated clusters in {end - start} seconds.")
    
    return clusters