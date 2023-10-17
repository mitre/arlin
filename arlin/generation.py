import logging
import time
from typing import List, Optional, Tuple

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


def _select_key_data(
    dataset: XRLDataset, keys: List[str], indices: np.array
) -> List[np.ndarray]:
    """Select data to cluster on from the dataset.

    Args:
        dataset (XRLDataset): XRLDataset to pull data from
        keys (List[str]): Keys for data within the dataset to pull
        indices (np.array): Indices of datapoints to use

    Returns:
        List[np.ndarray]: Collection of data to concatenate and cluster on
    """
    key_data = []
    for key in keys:
        val = getattr(dataset, key)[indices]

        if len(val.shape) != 2:
            val = np.expand_dims(val, axis=-1)

        key_data.append(val)

    return key_data


def _get_cluster_ons(
    dataset: XRLDataset,
    start_cluster_keys: List[str],
    intermediate_cluster_keys: List[str],
    term_cluster_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the data that we want to cluster on for initial, intermediate, and terminal.

    Args:
        dataset (XRLDataset): XRLDataset with the data to cluster on.
        start_cluster_keys (List[str]): Keys to cluster initial states on
        intermediate_cluster_keys (List[str]): Keys to cluster intermediate states on
        term_cluster_keys (List[str]): keys to cluster terminal states on

    Returns:
        Tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        Data to cluster for initial states, Data to cluster for intermediate states,
        data to cluster for terminal states, mask to identify intermediate states
    """

    mid_mask = np.ones([len(dataset.terminateds)], dtype=bool)
    mid_mask[dataset.start_indices] = False
    mid_mask[dataset.term_indices] = False

    start_data = _select_key_data(dataset, start_cluster_keys, dataset.start_indices)
    mid_data = _select_key_data(dataset, intermediate_cluster_keys, mid_mask)
    term_data = _select_key_data(dataset, term_cluster_keys, dataset.term_indices)

    cluster_on_start = np.concatenate(start_data, axis=-1)
    cluster_on_mid = np.concatenate(mid_data, axis=-1)
    cluster_on_term = np.concatenate(term_data, axis=-1)

    return cluster_on_start, cluster_on_mid, cluster_on_term, mid_mask


def generate_clusters(
    dataset: XRLDataset,
    start_cluster_keys: List[str],
    intermediate_cluster_keys: List[str],
    term_cluster_keys: List[str],
    num_clusters: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, object, object, object]:
    """Generate clusters from the given XRLDataset.

    Args:
        dataset (XRLDataset): XRLDataset to cluster on.
        start_cluster_keys (List[str]): Keys to cluster initial states on
        intermediate_cluster_keys (List[str]): Keys to cluster intermediate states on
        term_cluster_keys (List[str]): keys to cluster terminal states on
        num_clusters (int): Number of intermediate clusters to find in intermediate
            (not intitial or terminal) states
        seed (Optional[int], optional): Seed for clustering. Defaults to None.

    Raises:
        ValueError: No initial states found.
        ValueError: No terminal states found.
        ValueError: Not enough datapoints given (< num_clusters)

    Returns:
        Tuple(np.ndarray, object, object, object):
        Cluster values for each datapoint, initial cluster estimator, intermediate cluster
        estimator, terminal cluster estimator
    """
    logging.info(f"Generating {num_clusters} clusters.")

    start = time.time()

    (cluster_on_start, cluster_on_mid, cluster_on_term, mid_mask) = _get_cluster_ons(
        dataset, start_cluster_keys, intermediate_cluster_keys, term_cluster_keys
    )

    if len(cluster_on_start) == 0:
        raise ValueError("No initial indices found! Cancelling clustering.")
    else:
        start_algo = MeanShift()
        start_clusters = start_algo.fit(cluster_on_start)
        start_clusters = start_clusters.labels_

    if len(cluster_on_term) == 0:
        raise ValueError("No terminal indices found! Cancelling clustering.")
    else:
        term_algo = MeanShift()
        term_clusters = term_algo.fit(cluster_on_term)
        term_clusters = term_clusters.labels_

    if num_clusters > len(cluster_on_mid):
        raise ValueError(
            f"Not enough datapoints {len(cluster_on_mid)} to create \
                {num_clusters} clusters."
        )

    mid_algo = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
    mid_clusters = mid_algo.fit(cluster_on_mid)
    mid_clusters = mid_clusters.labels_

    n_start_clusters = len(set(start_clusters))

    start_clusters = np.array([x + num_clusters for x in start_clusters], dtype=int)
    term_clusters = np.array(
        [x + n_start_clusters + num_clusters for x in term_clusters], dtype=int
    )

    clusters = np.empty([len(dataset.terminateds)], dtype=int)
    clusters[mid_mask] = mid_clusters
    clusters[dataset.start_indices] = start_clusters
    clusters[dataset.term_indices] = term_clusters

    end = time.time()
    logging.info(f"\tSuccessfully generated clusters in {end - start} seconds.")

    return clusters, start_algo, mid_algo, term_algo
