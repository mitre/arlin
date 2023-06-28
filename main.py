import argparse
import yaml
from typing import Any, Dict
import logging
import os
import warnings
import gym

import arlin.data_analysis.latent_analysis as latent_analysis
import arlin.data_analysis.analytics_graphing as analytics_graphing
from arlin.data_analysis.graphers import ClusterGrapher, LatentGrapher
import arlin.dataset_creation.dataset_creator as dataset_creator
from arlin.data_analysis.xrl_dataset import XRLDataset
from arlin.data_analysis.samdp import SAMDP
import arlin.utils.data_analysis_utils as da_utils
    
def dataset_creation():
    model = dataset_creator.load_hf_sb_model(repo_id='sb3/ppo-LunarLander-v2',
                                              filename='ppo-LunarLander-v2.zip',
                                              sb_algo_str='ppo')
    env = gym.make('LunarLander-v2')
    
    datapoints = dataset_creator.collect_datapoints(model=model,
                                                   algo_str='ppo',
                                                   env=env,
                                                   num_datapoints=50000,
                                                   random=False)
    
    dataset_creator.save_datapoints(datapoint_dict=datapoints, 
                                    file_path='/nfs/lslab2/arlin/data_zoo/LunarLander-v2/ppo-50000.pkl')
        

def get_data(load_embeddings=False, load_clusters=False):
    dataset = XRLDataset(dataset_path='/nfs/lslab2/arlin/data_zoo/LunarLander-v2/ppo-50000.pkl')
    
    if load_embeddings:
        embeddings = da_utils.load_data(file_path='/nfs/lslab2/arlin/data_zoo/LunarLander-v2/embeddings/ppo-50000/latent_actors/test_embeddings.pkl')
    else:
        embeddings = latent_analysis.generate_embeddings(dataset=dataset,
                                                        activation_key='latent_actors',
                                                        perplexity=500,
                                                        n_train_iter=4000,
                                                        output_dim=2,
                                                        seed=12345)
        
        da_utils.save_data(data=embeddings, 
                        file_path='/nfs/lslab2/arlin/data_zoo/LunarLander-v2/embeddings/ppo-50000/latent_actors/test_embeddings.pkl')  
    
    if load_clusters: 
        clusters = da_utils.load_data(file_path='/nfs/lslab2/arlin/data_zoo/LunarLander-v2/clusters/ppo-50000/test_clusters_hac.pkl')
    else:
        clusters = latent_analysis.generate_clusters(cluster_on=dataset.latent_actors,
                                                     num_clusters=16,
                                                     clustering_method='hac')
        
        da_utils.save_data(data=clusters,
                        file_path='/nfs/lslab2/arlin/data_zoo/LunarLander-v2/clusters/ppo-50000/test_clusters_hac.pkl')
    
    return embeddings, clusters
        
def graph_latent_analytics(embeddings, clusters, dataset):
    grapher = LatentGrapher(embeddings, dataset)
    
    embeddings_data = grapher.embeddings_graph_data()
    cluster_data = grapher.clusters_graph_data(clusters)
    db_data = grapher.decision_boundary_graph_data()
    init_term_data = grapher.initial_terminal_state_data()
    ep_prog_data = grapher.episode_prog_graph_data()
    conf_data = grapher.confidence_data()
    
    base_path = './outputs/latent_analytics_hac/'
    for data in [#(embeddings_data, "embeddings.png"),
                 (cluster_data, "clusters.png"),
                 #(db_data, "decision_boundaries.png"),
                 #(init_term_data, "initial_terminal.png"),
                 #(ep_prog_data, "episode_progression.png"),
                 #(conf_data, "confidence.png")
                 ]:
        path = os.path.join(base_path, data[1])
        
        analytics_graphing.graph_individual_data(data[0], path)
    
    #analytics_graphing.graph_multiple_data('Latent Analytics', [db_data, conf_data, cluster_data, ep_prog_data], './outputs/combined_graphs_hac/latent_analytics.png')

def graph_cluster_analytics(dataset, clusters):
    grapher = ClusterGrapher(dataset, clusters)
    
    cluster_conf = grapher.cluster_confidence()
    cluster_rewards = grapher.cluster_rewards()
    
    base_path = './outputs/cluster_analytics_hac/'
    for data in [[cluster_conf, 'cluster_confidence.png'], [cluster_rewards, 'cluster_rewards.png']]:
        path = os.path.join(base_path, data[1])
        analytics_graphing.graph_individual_data(data[0], path)
        
    analytics_graphing.graph_multiple_data('Cluster Analytics', [cluster_rewards, cluster_conf], './outputs/combined_graphs/cluster_analytics_hac.png')

def samdp(clusters, dataset):
    samdp = SAMDP(clusters, dataset)
    # complete_graph = samdp.save_complete_graph('./outputs/samdp/samdp_complete.png')
    # et_graph = samdp.save_early_termination_paths('./outputs/samdp/samdp_et.png')
    likely_graph = samdp.save_likely_paths('./outputs/samdp/samdp_likely.png')
    #simplified_graph = samdp.save_simplified_graph('./outputs/samdp/samdp_simplified.png')
    samdp.save_paths(likely_graph, 3, 7, './outputs/samdp/samdp_path_3_7.png')
    samdp.save_paths(likely_graph, 3, 7, './outputs/samdp/samdp_path_3_7_bp.png', best_path_only=True)
    #samdp.save_txt('./outputs/samdp/samdp.txt')

def main(args) -> None:
    #dataset_creation()
    embeddings, clusters = get_data(load_embeddings=args.load_embeddings,
                                    load_clusters=args.load_clusters)
    
    dataset = XRLDataset(dataset_path='/nfs/lslab2/arlin/data_zoo/LunarLander-v2/ppo-50000.pkl')
    
    graph_latent_analytics(embeddings, clusters, dataset)
    graph_cluster_analytics(dataset, clusters)
    #samdp(clusters, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_embeddings', action='store_true')
    parser.add_argument('--load_clusters', action='store_true')
    args = parser.parse_args()
    
    # Logging and warning setup
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    main(args)
    