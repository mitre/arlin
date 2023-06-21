import argparse
import yaml
from typing import Any, Dict
import logging
import os
import warnings

from arlin.dataset_creation.dataset_creator import DatasetCreator
#from arlin.data_analysis.latent_analyzer import LatentAnalyzer

import arlin.data_analysis.latent_analysis as latent_analysis
import arlin.data_analysis.analytics_graphing as analytics_graphing
from arlin.data_analysis.xrl_dataset import XRLDataset
from arlin.data_analysis.samdp import SAMDP
import arlin.utils.data_analysis_utils as da_utils

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
        params = config['DATA_ANALYSIS']
        dataset = XRLDataset(params['dataset_path'])
        
        base_path = '/nfs/lslab2/arlin/data_zoo/LunarLander-v2/'
        
        if params['load_embeddings'] is None:
            embeddings = latent_analysis.generate_embeddings(dataset, 
                                                            **params['EMBEDDINGS'])
            da_utils.save_data(embeddings, os.path.join(base_path, 'embeddings', 'ppo-50000', 'latent_actors', 'test_embeddings.pkl'))
        else:
            embeddings = da_utils.load_data(params['load_embeddings'])
            
        if params['load_clusters'] is None:  
            clusters = latent_analysis.generate_clusters(dataset, embeddings=embeddings,
                                                        **params['CLUSTERS'])
            da_utils.save_data(clusters, os.path.join(base_path, 'clusters', 'ppo-50000', 'test_clusters.pkl'))
        else:
            clusters = da_utils.load_data(params['load_clusters'])
        
        # grapher = analytics_graphing.LatentGraphData(embeddings, dataset)
        
        # embeddings_data = grapher.embeddings_graph_data()
        # cluster_data = grapher.clusters_graph_data(clusters)
        # db_data = grapher.decision_boundary_graph_data()
        # init_term_data = grapher.initial_terminal_state_data()
        # ep_prog_data = grapher.episode_prog_graph_data()
        # conf_data = grapher.confidence_data()
        
        # base_path = './outputs/individual_graphs/'
        # for data in [(embeddings_data, "embeddings.png"),
        #              (cluster_data, "clusters.png"),
        #              (db_data, "decision_boundaries.png"),
        #              (init_term_data, "initial_terminal.png"),
        #              (ep_prog_data, "episode_progression.png"),
        #              (conf_data, "confidence.png")]:
        #     path = os.path.join(base_path, data[1])
            
        #     analytics_graphing.graph_individual_data(data[0], path)
        
        # analytics_graphing.graph_multiple_data('Analytics', [db_data, cluster_data, ep_prog_data], './outputs/combined_graphs/analytics.png')
        #analytics_graphing.graph_cluster_analysis(dataset, clusters, './outputs/analysis/analytics.png')
        
        samdp = SAMDP(clusters, dataset)
        complete_graph = samdp.save_complete_graph('./outputs/samdp/samdp_complete.png')
        # et_graph = samdp.save_early_termination_paths('./outputs/samdp/samdp_et.png')
        likely_graph = samdp.save_likely_paths('./outputs/samdp/samdp_likely.png')
        samdp.save_paths(likely_graph, 15, 3, './outputs/samdp/samdp_path_3_7.png')
        #samdp.save_txt('./outputs/samdp/samdp.txt')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='Path to main config.')
    args = parser.parse_args()
    config = get_config(args.config)
    
    # Logging and warning setup
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    main(config)
    