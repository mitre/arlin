import numpy as np
from typing import Any
import os
import logging


def save_data(data: Any, file_path: str) -> None:
    """Save data as a pickle file to given save path.
    
    Args:
        - data (Any): Data to save
        - file_path (str): File path to save the data to
    """
    
    if not file_path[-4:] == '.npy':
            file_path += '.npy'
            
    logging.info(f"Saving data to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    np.save(file_path, data=data)
    
    logging.info("\tData saved successfully.")

def load_data(file_path: str) -> Any:
    """Load and return data from given file path:
    
    Args:
        - file_path (str): Path to load file from
        
    Returns:
        - Any: Loaded data
    """
    
    if not file_path[-4:] == '.npy':
            raise ValueError("Can only load .npy files")
            
    logging.info(f"Loading data from {file_path}...")
    data = np.load(file_path)
    
    logging.info("\tData loaded successfully.")
    
    return data
