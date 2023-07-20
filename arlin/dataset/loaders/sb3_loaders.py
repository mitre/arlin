import arlin.utils as utils
import logging

from stable_baselines3.common.base_class import BaseAlgorithm
from huggingface_sb3 import load_from_hub

def load_hf_sb_model(repo_id: str,
                     filename: str,
                     algo_str: str) -> BaseAlgorithm:
    """Load a stable-baselines3 model from huggingface.
    
    Args:
        - repo_id (str): Repo_ID where the model is stored on huggingface
        - filename (str): Filename of the model zip within the repo on huggingface
    
    Returns:
        - BaseAlgorithm: Trained SB3 model
    """
    logging.info(f"Loading model {repo_id}/{filename} from huggingface...")
    try:
        checkpoint_path = load_from_hub(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise ValueError(f"Model could not be loaded from huggingface.\n{e}")
    
    model = load_sb_model(checkpoint_path, algo_str)
    
    return model

def load_sb_model(path: str, algo_str: str) -> BaseAlgorithm:
    """Load a stable-baselines3 model from a given path.

    Args:
        path (str): Path to the SB3 trained model zip.
        algo_str (str): Algorithm that was used to train the model.

    Returns:
        BaseAlgorithm: Trained SB3 model
    """
    algorithm = utils.get_sb3_algo(algo_str.lower())
    model = algorithm.load(path)
    
    return model