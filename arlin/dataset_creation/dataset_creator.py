from typing import Union, Dict, Any
import os
import gym
import arlin.utils.dataset_creation_utils as utils
from tqdm import tqdm
import pickle
import logging

from huggingface_sb3 import load_from_hub

class DatasetCreator():
    
    """
    #### Description:
    The DatasetCreator class is used to create an XRL dataset from a given model or a
    trained model loaded from huggingface.com. All models need to be trained using
    stable-baselines3.
    
    Args:
        - repo_id (str): Repo_ID where the model is stored on huggingface
        - filename (str): Filename of the model within the repo on huggingface 
        - algorithm (str): Algorithm name that the model was trained with
        - environment (str): OpenAI Gym registered env name
        - save_dir (str): Path to the directory in which to save the XRL dataset
        - num_episodes (int): Number of episodes to save into dataset
        - load_path (Union[str, None]): Optional path to a trained models pkl file. If
            None, model is loaded from huggingface.com

    """
    
    def __init__(
        self, 
        repo_id: str,
        filename: str,
        algorithm: str, 
        environment: str,
        save_dir: str,
        load_path: Union[str, None] = None
        ):
        self.algo_str = algorithm.lower()
        self.env_str = environment
        
        self.algorithm = utils.get_algo(self.algo_str.lower())
        self.env = gym.make(self.env_str)
        self.save_dir = os.path.join(save_dir, self.env_str)
        self.load_path = load_path
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        if self.load_path is None:
            self.load_path = self._load_hf_model(repo_id, filename)
        
        self.model = self.algorithm.load(self.load_path)
        self.gatherer = utils.get_dataset_gatherer(self.algo_str)(self.model.policy)
        
    def _load_hf_model(self, repo_id: str, filename: str) -> str:
        """
        Return the model path for the local model downloaded from huggingface for the
        given algorithm and environment.
        
        Args:
            - repo_id (str): Repo_ID where the model is stored on huggingface
            - filename (str): Filename of the model within the repo on huggingface
        
        Returns:
            - str: Path to the local model downloaded from huggingface
        """
        logging.info(f"Loading model {repo_id}/{filename} from huggingface...")
        try:
            checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
        except Exception as e:
            logging.error(f"Model could not be loaded from huggingface.\n{e}")
        return checkpoint
    
    def collect_episodes(self, num_episodes: int, random: bool = False) -> Dict[str, Any]:
        """
        Collect episodes needed for an XRL dataset.
        
        Args:
            - num_episodes (int): Number of episodes to collect.
        
        Returns:
            - Dict[str, Any]: Dictionary of datapoints after num_episodes
        """
        
        logging.info("Collecting episodes...")
        datapoint_dict = utils.get_datapoint_dict(self.algo_str)()
        
        for _ in tqdm(range(num_episodes)):
            obs = self.env.reset()
            step = 0
            done = False
            
            while not done:
                action = self.gatherer.gather_data(obs, datapoint_dict)
                
                if random:
                    action = self.env.action_space.sample()
                
                obs, reward, done, _ = self.env.step(action)
                datapoint_dict.add_base_data(obs, action, reward, done, step)
                step += 1
        
        return datapoint_dict.get_dict()
    
    def collect_datapoints(self, num_datapoints: int, random: bool = False) -> Dict[str, Any]:
        """
        Collect datapoints needed for an XRL dataset.
        
        Args:
            - num_datapoints (int): Number of datapoints to collect.
        
        Returns:
            - Dict[str, Any]: Dictionary of num_datapoints datapoints
        """
        
        logging.info("Collecting datapoints...")
        datapoint_dict = utils.get_datapoint_dict(self.algo_str)()
        
        obs = self.env.reset()
        step = 0
        for _ in tqdm(range(num_datapoints)):
            action = self.gatherer.gather_data(obs, datapoint_dict)
            
            if random:
                action = self.env.action_space.sample()
                
            obs, reward, done, _ = self.env.step(action)
            datapoint_dict.add_base_data(obs, action, reward, done, step)
            step += 1
            
            if done:
                obs = self.env.reset()
                step = 0
                done = False
        
        return datapoint_dict.get_dict()
    
    def save_datapoints(self, datapoint_dict: Dict[str, Any]) -> None:
        """
        Save dictionary of datapoints to self.save_dir.
        
        Args:
            - datapoint_dict (Dict[str, Any]): Dictionary of XRL datapoints to save
        """
        num_points = datapoint_dict['actions'].shape[0]
        filename = f"{self.algo_str}-{num_points}.pkl"
        logging.info(f"Saving datapoints to {self.save_dir}/{filename}...")
        with open(os.path.join(self.save_dir, filename), 'wb') as handle:
            pickle.dump(datapoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)