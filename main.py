import argparse
import yaml
from typing import Any, Dict
import logging
import os
import warnings

from perfect_timing.dataset_creation.dataset_creator import DatasetCreator
from perfect_timing.data_analysis.data_analyzer import DataAnalyzer

def get_config(config_path: str) -> Dict[str, Any]:
    """
    Load the YAML config file from the given path.

    Args:
        config_path (str): Path to load YAML config file from.

    Returns:
        Dict[str, Any]: Dictionary version of YAML config file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config: Dict[str, Any]) -> None:
    """
    Load a model and collect XRL datapoints.
    
    Args:
        - config (Dict[str, Any]): Config file with necessary information for running
    """ 
    
    if config['dataset_creation']:
        # Create the DatasetCreator
        params = config['DATASET_CREATION']
        dataset_creator = DatasetCreator(**params['DATASET_CREATOR'])
        
        # Collect data
        if params['collect_episodes']:
            datapoints = dataset_creator.collect_episodes(params['num_collect'])
        else:
            datapoints = dataset_creator.collect_datapoints(params['num_collect'])
        
        # Save results
        dataset_creator.save_datapoints(datapoints)
    
    if config['data_analysis']:
        # Create the DataAnalyzer
        params = config['DATA_ANALYSIS']
        analyzer = DataAnalyzer()
        analyzer.load_data(params['dataset_path'])
        embeddings = analyzer.get_embeddings(**params['EMBEDDINGS'])
        analyzer.graph_action_embeddings()
        clusters = analyzer.get_clusters(**params['CLUSTERS'])
        analyzer.graph_clusters()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to main config.')
    args = parser.parse_args()
    config = get_config(args.config)
    
    # Logging and warning setup
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    main(config)
    