import pickle
import numpy as np
import logging

class XRLDataset():
    
    def __init__(self, dataset_path: str):
        dataset_file = open(dataset_path,'rb')
        dataset = pickle.load(dataset_file)
        dataset_file.close()
        
        required_keys = ['observations', 'actions', 'rewards', 'dones']
        for key in required_keys:
            if key not in dataset:
                raise ValueError('Dataset must include, at minimum, the keys '\
                    f'{required_keys}: Missing {key}.')
        
        for key in dataset:
            setattr(self, key, dataset[key])
        
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        logging.info('Extracting necessary additional data from dataset.')
        logging.info("\tSetting self.num_datapoints.")
        self.num_datapoints = len(self.observations)
        self._set_episode_prog_indices()
        self._set_distinct_state_data()
        logging.info('Done setting dataset analysis variables.')
    
    def _set_episode_prog_indices(self):
        """Extract episode start and termination indices from the dataset.
        """
        
        logging.info('\tSetting self.done_indices.')
        logging.info('\tSetting self.start_indices.')
        
        done_indices = np.where(self.dones == 1)[0]
        # Start indices are always the first after a 'done' flag
        start_indices = done_indices + 1
        # Add the intitial start index
        start_indices = np.insert(start_indices, 0, 0)
        
        # Remove extra start index if the last datapoint is terminal
        if start_indices[-1] == self.num_datapoints:
            start_indices = start_indices[:-1]
        
        self.done_indices = done_indices
        self.start_indices = start_indices
    
    def _set_distinct_state_data(self):
        """Extract the unique state indices and corresponding state mapping to identify
        unique observations in the dataset. T-SNE has trouble with duplicate states so
        mapping unique states together is beneficial.
        """
        
        logging.info('\tSetting self.unique_state_indices.')
        logging.info('\tSetting self.state_mapping.')
        
        outputs = np.unique(
            self.observations, 
            return_index=True, 
            return_inverse=True, 
            axis=0)
        
        _, unique_state_indices, state_mapping = outputs
        self.unique_state_indices = unique_state_indices
        self.state_mapping = state_mapping