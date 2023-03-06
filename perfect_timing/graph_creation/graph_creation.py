from perfect_timing.graph_creation.graph_data import GraphData

from typing import Any, Dict, Union
from sklearn.manifold import TSNE
import numpy as np
import logging
import pickle
import os

class GraphCreator():
    def __init__(
        self, 
        load_embeddings: bool,
        num_components: int,
        perplexity: int,
        tsne_init: str,
        tsne_seed: int,
        GRAPH_DATA: Dict[str, Any]
        ):
        self.graph_data = GraphData(**GRAPH_DATA)
        
        dataset_dir = os.path.dirname(self.graph_data.dataset_path)
        embeddings_dir = os.path.join(dataset_dir, "embeddings")
        
        if load_embeddings:
            filename = os.path.basename(self.graph_data.dataset_path).split('.')[0]
            filename = f"{filename}-{self.graph_data.activation_key}_embeddings.pkl"
            file_path = os.path.join(embeddings_dir, filename)
            logging.info(f'Loading embeddings from {file_path}')
            embeddings_file = open(file_path,'rb')
            embeddings = pickle.load(embeddings_file)
            embeddings_file.close()
        else:
            embeddings = self._get_embeddings(
                num_components, 
                perplexity, 
                tsne_init, 
                tsne_seed)
            
            if not os.path.exists(embeddings_dir):
                os.mkdir(embeddings_dir)
            self.save_embeddings(embeddings, embeddings_dir)

            
    def _get_embeddings(
        self,
        num_components: int,
        perplexity: int,
        tsne_init: str, 
        tsne_seed: int
        ) -> None:
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

    def save_embeddings(self, embeddings: np.ndarray, embeddings_save_dir: str) -> None:
        """
        Save activation embeddings.
        
        Args:
            - embeddings_save_dir (str): Directory to save the embeddings into
            - embeddings (np.ndarray): numpy array of activation embeddings from T-SNE
        """
        
        dataset_file = os.path.basename(self.graph_data.dataset_path).split('.')[0]
        filename = f'{dataset_file}-{self.graph_data.activation_key}_embeddings.pkl'
        logging.info(f"Saving embeddings to {embeddings_save_dir}/{filename}...")
        
        with open(os.path.join(embeddings_save_dir, filename), 'wb') as handle:
            pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)