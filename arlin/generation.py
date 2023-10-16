import logging
import time
from typing import Optional, Tuple

import numpy as np

# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans, MeanShift

from arlin.dataset.xrl_dataset import XRLDataset


def generate_embeddings(
    dataset: XRLDataset,
    activation_key: str,
    perplexity: int,
    n_train_iter: int,
    output_dim: int = 2,
    seed: int = 12345,
) -> np.ndarray:
    """Generate TSNE embeddings from the given XRLDataset.

    Args:
        dataset (XRLDataset): XRLDataset generated from an RL policy.
        activation_key (str): Data that we want to embed on.
        perplexity (int): Perplexity value for TSNE
        n_train_iter (int): Number of training iterations for TSNE
        output_dim (int, optional): Output dimensions of the embeddings. Defaults to 2.
        seed (int, optional): Seed for TSNE. Defaults to 12345.

    Returns:
        np.ndarray: TSNE embeddings
    """
    logging.info(f"Generating embeddings from dataset.{activation_key}.")

    start = time.time()

    embedder = TSNE(
        n_jobs=4,
        n_components=output_dim,
        perplexity=perplexity,
        n_iter=n_train_iter,
        verbose=1,
        random_state=seed,
    )

    activations = getattr(dataset, activation_key)
    unique_activations = activations[dataset.unique_state_indices]

    embeddings = embedder.fit_transform(unique_activations)
    embeddings = [embeddings[index] for index in dataset.state_mapping]

    end = time.time()

    logging.info(f"\tSuccessfully generated embeddings in {(end - start) % 60} minutes.")

    return np.array(embeddings)


def _get_cluster_ons(
    dataset: XRLDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the data that we want to cluster on for initial, intermediate, and terminal.

    Args:
        dataset (XRLDataset): XRLDataset with the data to cluster on.

    Returns:
        Tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        Data to cluster for intermediate states, mask to identify intermediate states,
        Data to cluster for initial states, data to cluster for terminal states
    """
    start_latents = dataset.latent_actors[dataset.start_indices]
    term_latents = dataset.latent_actors[dataset.term_indices]

    start_values = np.expand_dims(dataset.critic_values[dataset.start_indices], axis=-1)
    start_rewards = np.expand_dims(dataset.rewards[dataset.start_indices], axis=-1)
    term_values = np.expand_dims(dataset.critic_values[dataset.term_indices], axis=-1)
    term_rewards = np.expand_dims(dataset.rewards[dataset.term_indices], axis=-1)
    term_total_rewards = np.expand_dims(
        dataset.total_rewards[dataset.term_indices], axis=-1
    )

    mask = np.ones([len(dataset.terminateds)], dtype=bool)
    mask[dataset.start_indices] = False
    mask[dataset.term_indices] = False

    latents = dataset.latent_actors[mask]
    values = np.expand_dims(dataset.critic_values[mask], axis=-1)
    rewards = np.expand_dims(dataset.rewards[mask], axis=-1)

    cluster_on = np.concatenate(
        [
            latents,
            values,
            rewards,
        ],
        axis=-1,
    )

    cluster_on_start = np.concatenate(
        [start_latents, start_values, start_rewards], axis=-1
    )

    cluster_on_term = np.concatenate(
        [term_latents, term_total_rewards, term_rewards, term_values], axis=-1
    )

    return cluster_on, mask, cluster_on_start, cluster_on_term


def generate_clusters(
    dataset: XRLDataset, num_clusters: int, seed: Optional[int] = None
) -> Tuple[np.ndarray, object, object, object]:
    """Generate clusters from the given XRLDataset.

    Args:
        dataset (XRLDataset): XRLDataset to cluster on.
        num_clusters (int): Number of intermediate clusters to find.
        seed (Optional[int], optional): Seed for clustering. Defaults to None.

    Raises:
        ValueError: Not enough datapoints given (< num_clusters)

    Returns:
        Tuple(np.ndarray, object, object, object):
        Cluster values for each datapoint, initial cluster estimator, intermediate cluster
        estimator, terminal cluster estimator
    """
    logging.info(f"Generating {num_clusters} clusters.")

    start = time.time()

    (cluster_on, cluster_on_mask, cluster_on_start, cluster_on_term) = _get_cluster_ons(
        dataset
    )

    if len(cluster_on_start) == 0:
        logging.warning("No start indices found in dataset.")
        start_clusters = []
    else:
        start_algo = MeanShift()
        start_clusters = start_algo.fit(cluster_on_start)
        start_clusters = start_clusters.labels_

    if len(cluster_on_term) == 0:
        logging.warning("No terminal indices found in dataset.")
        term_clusters = []
    else:
        term_algo = MeanShift()
        term_clusters = term_algo.fit(cluster_on_term)
        term_clusters = term_clusters.labels_

    if num_clusters > len(cluster_on):
        raise ValueError(
            f"Not enough datapoints {len(cluster_on)} to create {num_clusters} clusters."
        )

    mid_algo = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
    mid_clusters = mid_algo.fit(cluster_on)
    mid_clusters = mid_clusters.labels_

    n_clusters = len(set(mid_clusters))
    n_start_clusters = len(set(start_clusters))

    start_clusters = np.array([x + n_clusters for x in start_clusters], dtype=int)
    term_clusters = np.array(
        [x + n_start_clusters + n_clusters for x in term_clusters], dtype=int
    )

    clusters = np.empty([len(dataset.terminateds)], dtype=int)
    clusters[cluster_on_mask] = mid_clusters
    clusters[dataset.start_indices] = start_clusters
    clusters[dataset.term_indices] = term_clusters

    end = time.time()
    logging.info(f"\tSuccessfully generated clusters in {end - start} seconds.")

    return clusters, start_algo, mid_algo, term_algo
