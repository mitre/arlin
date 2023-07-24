import pickle
import numpy as np
import logging
import os
import gymnasium as gym
from tqdm import tqdm
import dataclasses
from typing import Dict, List, Type, Tuple

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
    
    def fill(self, 
             num_datapoints: int = 50000) -> None:
        
        logging.info(f'Collecting {num_datapoints} datapoints.')
        collected_datapoints = 0
        num_episodes = 0
        while collected_datapoints < num_datapoints:
            datapoints, trunc = self.collect_episode()
            
            if trunc:
                logging.debug("\tSkipping episode due to truncation.")
                continue
            
            self._append_datapoints(datapoints)
            
            collected_datapoints += len(datapoints)
            num_episodes += 1

            logging.info(f"\tEpisode {num_episodes} |"\
                f" Collected: {len(datapoints)} |"\
                    f" Total: {collected_datapoints}.")
            
        self._append_datapoints(datapoints)
        
        logging.info(f'Collected {collected_datapoints} datapoints total.')
        if collected_datapoints > num_datapoints:
            num_extra = collected_datapoints - num_datapoints
            logging.debug(f'{num_extra} datapoints have been collected"\
                " for cleaner MDP creation.')

    def collect_episode(self) -> Tuple[List[Type[BaseDatapoint]], bool]:
        ep_datapoints = []
        obs, _ = self.env.reset()
        step = 0
        while True:
            datapoint, action =  self.collector.collect_internal_data(observation=obs)
                
            new_obs, reward, term, trunc, _ = self.env.step(action)
            datapoint.add_base_data(obs, action, reward, term, trunc, step)
            ep_datapoints.append(datapoint)
            step += 1
            obs = new_obs
            
            if term or trunc:
                break
        
        return ep_datapoints, trunc

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
            
            cur_value = getattr(self, field_name)
            new_data = np.array(data_dict[field_name])
            
            if cur_value.size == 0:
                setattr(self, field_name, new_data)
            else:
                updated_value = np.concatenate([cur_value, new_data]) 
                setattr(self, field_name, updated_value)
    
    def analyze_dataset(self):
        logging.info('Extracting necessary additional data from dataset.')
        logging.info("\tSetting self.num_datapoints.")
        self.num_datapoints = len(self.observations)
        self._set_total_rewards()
        self._set_episode_prog_indices()
        self._normalize_steps()
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
            
            if self.terminateds[i] or self.truncateds[0]:
                cur_total = 0
        
        self.total_rewards = np.array(total_rewards)
    
    def _set_episode_prog_indices(self):
        """Extract episode start and termination indices from the dataset.
        """
        
        logging.info('\tSetting self.start_indices.')
        logging.info('\tSetting self.term_indices.')
        logging.info('\tSetting self.trunc_indices.')
        
        start_indices = np.where(self.steps == 0)[0]
        term_indices = np.where(self.terminateds == 1)[0]
        trunc_indices = np.where(self.truncateds == 1)[0]
        
        self.start_indices = start_indices
        self.term_indices = term_indices
        self.trunc_indices = trunc_indices
        
        if len(self.start_indices) == 0:
            logging.warning('No start indices identified.')
        
        if len(self.term_indices) == 0:
            logging.warning('No terminated indices identified.')
            
        if len(self.trunc_indices) == 0:
            logging.warning('No truncated indices identified.')
    
    def _normalize_steps(self):
        logging.info("\tNormalizing self.steps.")
        self.steps = self.steps.astype('float')
        
        key_indices = list(self.start_indices) + [len(self.terminateds)]
        for i in range(len(key_indices) - 1):
            ep_len = key_indices[i+1] - key_indices[i]
            
            for j in range(ep_len):
                idx = key_indices[i] + j
                
                self.steps[idx] = self.steps[idx] / ep_len
    
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
            out_dict['start_indices'] = self.start_indices
            out_dict['term_indices'] = self.term_indices
            out_dict['trunc_indices'] = self.trunc_indices
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