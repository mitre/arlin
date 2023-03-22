import pickle
from typing import Any
import os
import logging

CLUSTER_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#101010', '#6f2fff', 
                  '#0f8f7f', '#c48c5c', '#cf0fcf', '#4b0082')


def save_data(data: Any, save_dir: str, filename: str) -> None:
    """
    Save data as a pickle file to given save path.
    
    Args:
        - data (Any): Data to save
        - save_dir (str): Path to the directory to save the data to
        - filename (str): Filename for the data
    """
    file_path = os.path.join(save_dir, filename)
    logging.info(f"Saving data to {file_path}...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(load_dir: str, filename: str) -> Any:
    """
    Load and return data from given filename:
    
    Args:
        - load_dir (str): Directory to load file from
        - filename (str): Filename to load
        
    Returns:
        - Any: Loaded data
    """
    file_path = os.path.join(load_dir, filename)
    logging.info(f"Loading data from {file_path}...")
    data_file = open(file_path,'rb')
    data = pickle.load(data_file)
    data_file.close()
    
    return data