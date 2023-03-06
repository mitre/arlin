from abc import ABC
import pickle
import numpy as np
import logging

class GraphData(ABC):
    
    def __init__(self, dataset_path: str, activation_key: str):
        
        self.dataset_path = dataset_path
        self.activation_key = activation_key
        self._load_dataset(dataset_path)
        self._get_episode_data()
        self._get_distinct_state_info()
    
    def _load_dataset(self, dataset_path: str) -> None:
        dataset_file = open(dataset_path,'rb')
        dataset = pickle.load(dataset_file)
        dataset_file.close()
        
        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.rewards = dataset['rewards']
        self.dones = dataset['dones']
        
        assert self.activation_key in dataset.keys(), f"Activation Key not in dataset! Expects one of {dataset.keys()}"
        self.activations = dataset[self.activation_key]
    
    def _get_episode_data(self):
        self.done_indices = np.where(self.dones == 1)[0]
        self.start_indices = self.done_indices + 1
        self.start_indices = np.insert(self.start_indices, 0, 0)
        if self.start_indices[-1] == len(self.dones):
            self.start_indices = self.start_indices[:-1]
        
        self.final_states = self.observations[self.done_indices]
        self.start_states = self.observations[self.start_indices]
    
    def _get_distinct_state_info(self):
        self.unique_states, self.unique_indices, self.state_mapping = np.unique(self.observations, return_index=True, return_inverse=True, axis=0)
        logging.info(f"Found {self.unique_states.shape[0]} distinct states!")
        
    