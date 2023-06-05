import argparse
import yaml
from typing import Any, Dict
import logging
import os
import warnings

from arlin.dataset_creation.dataset_creator import DatasetCreator
from arlin.data_analysis.data_analyzer import DataAnalyzer

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
            datapoints = dataset_creator.collect_episodes(params['num_collect'],
                                                          params['random'])
        else:
            datapoints = dataset_creator.collect_datapoints(params['num_collect'],
                                                            params['random'])
        
        # Save results
        dataset_creator.save_datapoints(datapoints)
    
    if config['data_analysis']:
        # Create the DataAnalyzer
        params = config['DATA_ANALYSIS']
        analyzer = DataAnalyzer()
        analyzer.load_data(params['dataset_path'])
        
        embeddings = analyzer.get_embeddings(**params['EMBEDDINGS'])
        clusters = analyzer.get_clusters(**params['CLUSTERS'])
        
        # db_data = analyzer.decision_boundary_data()
        # cluster_data = analyzer.cluster_data()
        # cluster_confs = analyzer.analyze_clusters()
        # conf_data = analyzer.confidence_data()
        # state_data = analyzer.initial_terminal_state_data()
        # episode_prog_data = analyzer.episode_prog_data()
        # embed_data = analyzer.embeddings_data()
        
        # graph_data = [(db_data, "decision_boundaries.png"),
        #               (cluster_data, f"{analyzer.num_clusters}-clusters.png"),
        #               (conf_data, "confidence.png"),
        #               (state_data, "important_states.png"),
        #               (embed_data, "embeddings.png"),
        #               (episode_prog_data, "episode_prog.png"),
        #               (cluster_confs, "cluster_analysis.png")]
        
        # for i in graph_data:
        #     data, filename = i
        #     analyzer.graph_individual_data(data, filename)
        
        # analyzer.graph_analytics([cluster_data, conf_data], horizontal=True)
        samdp = analyzer.get_SAMDP()
        # analyzer.find_paths(3, 7)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to main config.')
    args = parser.parse_args()
    config = get_config(args.config)
    
    # Logging and warning setup
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    main(config)
    