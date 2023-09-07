import numpy as np
import logging
from arlin.dataset.xrl_dataset import XRLDataset
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans, MeanShift

from typing import Optional
import time

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
        n_jobs=4,
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
    
    logging.info(f"\tSuccessfully generated embeddings in {(end - start) % 60} minutes.")
    
    return np.array(embeddings)

def _get_cluster_ons(dataset: XRLDataset):
    cluster_on_start = dataset.critic_values[dataset.start_indices].reshape(-1, 1)
    cluster_on_term = dataset.total_rewards[dataset.term_indices].reshape(-1, 1)
    
    mask = np.ones([len(dataset.terminateds)], dtype=bool)
    mask[dataset.start_indices] = False
    mask[dataset.term_indices] = False
    
    latents = dataset.latent_actors[mask] 
    actions = np.expand_dims(dataset.actions[mask], axis=-1)
    values = np.expand_dims(dataset.critic_values[mask], axis=-1)
    # steps = np.expand_dims(dataset.steps[mask], axis=-1)
    rewards = np.expand_dims(dataset.rewards[mask], axis=-1)
    total_rewards = np.expand_dims(dataset.total_rewards[mask], axis=-1)
    confidences = np.expand_dims(np.amax(dataset.dist_probs, axis=1)[mask], axis=-1)
    
    cluster_on = np.concatenate([latents,
                                 actions,
                                 values,
                                 rewards,
                                 total_rewards,
                                 confidences], axis=-1)
    
    return cluster_on, mask, cluster_on_start, cluster_on_term

def generate_clusters(
    dataset: XRLDataset,
    num_clusters: int,
    seed: Optional[int] = None
) -> np.ndarray:
    
    logging.info(f"Generating {num_clusters} clusters.")
    
    start = time.time()
    
    (cluster_on, 
     cluster_on_mask, 
     cluster_on_start, 
     cluster_on_term) = _get_cluster_ons(dataset)
    
    if len(cluster_on_start) == 0:
        logging.warning('No start indices found in dataset.')
        start_clusters = []
    else:
        start_algo = MeanShift()
        start_clusters = start_algo.fit(cluster_on_start)
        start_clusters = start_clusters.labels_
    
    if len(cluster_on_term) == 0:
        logging.warning('No terminal indices found in dataset.')
        term_clusters = []
    else:
        term_algo = MeanShift()
        term_clusters = term_algo.fit(cluster_on_term)
        term_clusters = term_clusters.labels_
        
    if num_clusters > len(cluster_on):
        raise ValueError(f'Not enough datapoints {len(cluster_on)} to create {num_clusters} clusters.')
    
    mid_algo = KMeans(n_clusters=num_clusters,
                          random_state=seed, 
                          n_init='auto')
    mid_clusters = mid_algo.fit(cluster_on)
    mid_clusters = mid_clusters.labels_
    
    n_clusters = len(set(mid_clusters))
    n_start_clusters = len(set(start_clusters))
    
    start_clusters = np.array([x+n_clusters for x in start_clusters], dtype=int)
    term_clusters = np.array([x+n_start_clusters+n_clusters for x in term_clusters], dtype=int)
    
    clusters = np.empty([len(dataset.terminateds)], dtype=int)
    clusters[cluster_on_mask] = mid_clusters
    clusters[dataset.start_indices] = start_clusters
    clusters[dataset.term_indices] = term_clusters
    
    end = time.time()
    logging.info(f"\tSuccessfully generated clusters in {end - start} seconds.")
    
    return clusters, start_algo, term_algo, mid_algo