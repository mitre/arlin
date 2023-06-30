from typing import Union, Dict, Any
import os
import gym
import arlin.utils.dataset_creation_utils as utils
from tqdm import tqdm
import pickle
import logging

from huggingface_sb3 import load_from_hub
        
def load_hf_sb_model(repo_id: str,
                     filename: str,
                     algo_str: str) -> str:
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
        checkpoint_path = load_from_hub(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise ValueError(f"Model could not be loaded from huggingface.\n{e}")
    
    algorithm = utils.get_algo(algo_str.lower())
    model = algorithm.load(checkpoint_path)
    
    return model
    
def collect_episodes(model,
                     algo_str: str,
                     env: gym.Env,
                     num_episodes: int, 
                     random: bool = False
                     ) -> Dict[str, Any]:
    """
    Collect episodes needed for an XRL dataset.
    
    Args:
        - num_episodes (int): Number of episodes to collect.
    
    Returns:
        - Dict[str, Any]: Dictionary of datapoints after num_episodes
    """
    algo_str = algo_str.lower()
    logging.info("Collecting episodes...")
    datapoint_dict = utils.get_datapoint_dict(algo_str)()
    
    gatherer = utils.get_dataset_gatherer(algo_str)(model.policy)
    
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            action = gatherer.gather_data(obs, datapoint_dict)
            
            if random:
                action = env.action_space.sample()
            
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            datapoint_dict.add_base_data(obs, action, reward, total_reward, done, step)
            step += 1
    
    return datapoint_dict.get_dict()

def collect_datapoints(model,
                       algo_str: str,
                       env: gym.Env,
                       num_datapoints: int, 
                       random: bool = False
                       ) -> Dict[str, Any]:
    """
    Collect datapoints needed for an XRL dataset.
    
    Args:
        - num_datapoints (int): Number of datapoints to collect.
    
    Returns:
        - Dict[str, Any]: Dictionary of num_datapoints datapoints
    """
    algo_str = algo_str.lower()
    logging.info("Collecting datapoints...")
    datapoint_dict = utils.get_datapoint_dict(algo_str)()
    
    gatherer = utils.get_dataset_gatherer(algo_str)(model.policy)
    
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    for _ in tqdm(range(num_datapoints)):
        action = gatherer.gather_data(obs, datapoint_dict)
        
        if random:
            action = env.action_space.sample()
            
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        datapoint_dict.add_base_data(obs, action, reward, total_reward, done, step)
        step += 1
        
        if done:
            obs, _ = env.reset()
            total_reward = 0
            step = 0
            done = False
    
    return datapoint_dict.get_dict()

def save_datapoints(datapoint_dict: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary of datapoints to self.save_dir.
    
    Args:
        - datapoint_dict (Dict[str, Any]): Dictionary of XRL datapoints to save
    """
    
    if not file_path[-4:] == '.pkl':
        file_path += '.pkl'
    
    logging.info(f"Saving datapoints to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as handle:
        pickle.dump(datapoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)