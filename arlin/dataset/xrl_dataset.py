import pickle
import numpy as np
import logging
import os
import gymnasium as gym
from tqdm import tqdm
import dataclasses
from typing import Dict, List, Type

from arlin.dataset.collectors import BaseDataCollector, RandomDataCollector, BaseDatapoint


class XRLDataset():
    
    def __init__(self,
                 environment: gym.Env,
                 collector: BaseDataCollector = RandomDataCollector):
        
        self.env = environment
        self.collector = collector
        
        for field in dataclasses.fields(self.collector.datapoint_cls):
            if not hasattr(self, field.name):
                setattr(self, field.name, np.array([]))
        
        self.analyzed = False
    
    def fill(self, num_datapoints: int = 50000) -> None:
        obs, _ = self.env.reset()
        step = 0
        datapoints = []
        for _ in tqdm(range(num_datapoints)):
            datapoint, action =  self.collector.collect_internal_data(observation=obs)
                
            new_obs, reward, done, _, _ = self.env.step(action)
            datapoint.add_base_data(obs, action, reward, done, step)
            datapoints.append(datapoint)
            step += 1
            obs = new_obs
            
            if done:
                obs, _ = self.env.reset()
                step = 0
        
        self._append_datapoints(datapoints)

    def _append_datapoints(self, 
                                datapoints: List[Type[BaseDatapoint]]
                                ) -> Dict[str, np.ndarray]:
        
        field_names = [i.name for i in dataclasses.fields(self.collector.datapoint_cls)]
        
        data_dict = {i: [] for i in field_names}
        
        for i in range(len(datapoints)):
            datapoint = datapoints[i]
            
            for field_name in field_names:
                val = getattr(datapoint, field_name)
                data_dict[field_name].append(val)
        
        for field_name in field_names:
            setattr(self, field_name, np.array(data_dict[field_name]))
    
    
    def analyze_dataset(self):
        logging.info('Extracting necessary additional data from dataset.')
        logging.info("\tSetting self.num_datapoints.")
        self.num_datapoints = len(self.observations)
        self._set_total_rewards()
        self._set_episode_prog_indices()
        self._set_distinct_state_data()
        logging.info('Done setting dataset analysis variables.')
        self.analyzed = True
    
    def _set_total_rewards(self):
        logging.info("\tSetting self.total_rewards.")
        
        total_rewards = []
        
        cur_total = 0
        for i in range(self.num_datapoints):
            cur_total += self.rewards[i]
            total_rewards.append(cur_total)
            
            if self.dones[i]:
                cur_total = 0
        
        self.total_rewards = np.array(total_rewards)
    
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
        
        if len(self.start_indices) == 0:
            logging.warning('No start indices identified.')
        
        if len(self.done_indices) == 0:
            logging.warning('No terminal indices identified.')
    
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
    
    def get_dict(self) -> Dict[str, List[np.ndarray]]:
        
        out_dict = {}
        
        for field in dataclasses.fields(self.collector.datapoint_cls):
            out_dict[field.name] = np.array(getattr(self, field.name))
        
        if self.analyzed:
            out_dict['total_rewards'] = self.total_rewards
            out_dict['done_indices'] = self.done_indices
            out_dict['start_indices'] = self.start_indices
            out_dict['unique_state_indices'] = self.unique_state_indices
            out_dict['state_mapping'] = self.state_mapping
        
        return out_dict
    
    def save(self, file_path: str) -> None:
        """
        Save dictionary of datapoints to the given file_path.
        
        Args:
            - file_path str: Filepath to save XRL dataset to.
        """
        
        if not file_path[-4:] == '.pkl':
            file_path += '.pkl'
        
        logging.info(f"Saving datapoints to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as handle:
            pickle.dump(self.get_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load(self, load_path: str) -> None:
        dataset_file = open(load_path,'rb')
        dataset = pickle.load(dataset_file)
        dataset_file.close()
        
        for key in dataset:
            setattr(self, key, dataset[key])