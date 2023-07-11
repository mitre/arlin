import argparse
import yaml
from typing import Any, Dict
import logging
import os
import warnings
import gymnasium as gym

import arlin.data_analysis.latent_analysis as latent_analysis
import arlin.data_analysis.analytics_graphing as analytics_graphing
from arlin.data_analysis.graphers import ClusterGrapher, LatentGrapher
import arlin.dataset_creation.dataset_creator as dataset_creator
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

def simple_strike_setup():
    from simplestrike.adr import CurriculumManager
    from gymnasium.envs.registration import register
    from simplestrike.simple_strike import SimpleStrikeEnv
    
    register(
        id="SimpleStrike-v0",
        entry_point="simplestrike.simple_strike:SimpleStrikeEnv",
        reward_threshold=1.0,
        nondeterministic=True,
    )
    
    debug_config = "./simplestrike/config.yaml"
    with open(debug_config, "r") as file:
        cfg = yaml.safe_load(file)
        
    SCENARIO_CONFIG_PATH = cfg.get("SCENARIO_CONFIG_PATH")
    with open(SCENARIO_CONFIG_PATH, "r") as file:
        scenario_config = yaml.safe_load(file)
    cm = CurriculumManager(scenario_config, cfg["LESSON_PLAN"])
    lesson = cm.create_lesson(adr_lvls=(0, 0, 0, 0, 0, 0, 0))
    
    return {'lesson_config': lesson}

def dataset_creation(cfg: Dict[str, Any]):
    
    path = cfg['model_path']
    if path is None:
        model = dataset_creator.load_hf_sb_model(repo_id=f"sb3/{cfg['algo_str']}-{cfg['environment']}",
                                                filename=f"{cfg['algo_str']}-{cfg['environment']}.zip",
                                                algo_str=cfg['algo_str'])
    else:
        model = dataset_creator.load_sb_model(path=path, 
                                              algo_str=cfg['algo_str'])
    
    if cfg['environment'] == 'SimpleStrike-v0':
        env_cfg = simple_strike_setup()
    else:
        env_cfg = {}
        
    env = gym.make(cfg['environment'], **env_cfg)
    
    datapoints = dataset_creator.collect_datapoints(model=model,
                                                   algo_str=cfg['algo_str'],
                                                   env=env,
                                                   num_datapoints=cfg['num_datapoints'])
    
    save_path = f"/nfs/lslab2/arlin/data_zoo/{cfg['environment']}/{cfg['algo_str']}-{cfg['num_datapoints']}.pkl"
    dataset_creator.save_datapoints(datapoint_dict=datapoints, 
                                    file_path=save_path)
    
    return save_path

def get_data(cfg: Dict[str, Any],
             dataset, 
             load_embeddings=False, 
             load_clusters=False):
    
    embed_cfg = cfg['EMBEDDINGS']
    cluster_cfg = cfg['CLUSTERS']
    
    base_path = f"/nfs/lslab2/arlin/data_zoo/{cfg['environment']}"
    embeddings_path = f"{base_path}/embeddings/{cfg['algo_str']}-{cfg['num_datapoints']}/{embed_cfg['activation_key']}/embeddings.pkl"
    clusters_path = f"{base_path}/clusters/{cfg['algo_str']}-{cfg['num_datapoints']}/{cluster_cfg['num_clusters']}'.pkl"
    
    if load_embeddings:
        embeddings = da_utils.load_data(file_path=embeddings_path)
    else:
        embeddings = latent_analysis.generate_embeddings(dataset=dataset,
                                                        activation_key=embed_cfg['activation_key'],
                                                        perplexity=embed_cfg['perplexity'],
                                                        n_train_iter=embed_cfg['n_train_iter'],
                                                        output_dim=2,
                                                        seed=12345)
        
        da_utils.save_data(data=embeddings, file_path=embeddings_path)  
    
    if load_clusters: 
        clusters = da_utils.load_data(file_path=clusters_path)
    else:
        clusters = latent_analysis.generate_clusters(dataset=dataset,
                                                     embeddings=embeddings,
                                                     num_clusters=cluster_cfg['num_clusters'])
        
        da_utils.save_data(data=clusters, file_path=clusters_path)
    
    return embeddings, clusters
        
def graph_latent_analytics(run_dir:str, embeddings, clusters, dataset):
    grapher = LatentGrapher(embeddings, dataset)
    
    embeddings_data = grapher.embeddings_graph_data()
    cluster_data = grapher.clusters_graph_data(clusters)
    db_data = grapher.decision_boundary_graph_data()
    init_term_data = grapher.initial_terminal_state_data()
    ep_prog_data = grapher.episode_prog_graph_data()
    conf_data = grapher.confidence_data()
    
    base_path = os.path.join(run_dir, 'latent_analytics')
    for data in [(embeddings_data, "embeddings.png"),
                 (cluster_data, "clusters.png"),
                 (db_data, "decision_boundaries.png"),
                 (init_term_data, "initial_terminal.png"),
                 (ep_prog_data, "episode_progression.png"),
                 (conf_data, "confidence.png")
                 ]:
        path = os.path.join(base_path, data[1])
        
        analytics_graphing.graph_individual_data(path, data[0])
    
    combined_path = os.path.join(base_path, 'combined_analytics.png')
    analytics_graphing.graph_multiple_data(file_path=combined_path,
                                           figure_title='Latent Analytics', 
                                           graph_datas=[db_data, 
                                                        conf_data, 
                                                        cluster_data, 
                                                        ep_prog_data])

def graph_cluster_analytics(run_dir: str, dataset, clusters):
    grapher = ClusterGrapher(dataset, clusters)
    
    cluster_conf = grapher.cluster_confidence()
    cluster_rewards = grapher.cluster_rewards()
    cluster_values = grapher.cluster_values()
    
    base_path = os.path.join(run_dir, 'cluster_analytics')
    for data in [[cluster_conf, 'cluster_confidence.png'], 
                 [cluster_rewards, 'cluster_rewards.png'],
                 [cluster_values, 'cluster_values.png']
                 ]:
        path = os.path.join(base_path, data[1])
        analytics_graphing.graph_individual_data(path, data[0])
    
    combined_path = os.path.join(base_path, 'combined_analytics.png')
    analytics_graphing.graph_multiple_data(file_path=combined_path, 
                                           figure_title='Cluster Analytics', 
                                           graph_datas=[cluster_rewards, cluster_values])

def samdp(run_dir: str, cfg: Dict[str, Any], clusters, dataset):
    samdp = SAMDP(clusters, dataset)
    
    base_path = os.path.join(run_dir, 'samdp')
    
    complete_graph = samdp.save_complete_graph(f'{base_path}/samdp_complete.png')
    likely_graph = samdp.save_likely_paths(f'{base_path}/samdp_likely.png')
    simplified_graph = samdp.save_simplified_graph(f'{base_path}/samdp_simplified.png')
    
    path_path = os.path.join(base_path, f"samdp_path_{cfg['from_cluster']}_{cfg['to_cluster']}")
    
    # samdp.save_paths(cfg['from_cluster'], 
    #                  cfg['to_cluster'], 
    #                  f'{path_path}.png')
    
    # samdp.save_paths(cfg['from_cluster'], 
    #                  cfg['to_cluster'], 
    #                  f'{path_path}_verbose.png', 
    #                  verbose=True)
    
    # samdp.save_paths(cfg['from_cluster'], 
    #                  cfg['to_cluster'], 
    #                  f'{path_path}_bp.png', 
    #                  best_path_only=True)
    
    samdp.save_all_paths_to(cfg['to_cluster'], 
                            os.path.join(base_path, f"samdp_path_to_{cfg['to_cluster']}_verbose.png"),
                            verbose=True)
    
    samdp.save_all_paths_to(cfg['to_cluster'], 
                            os.path.join(base_path, f"samdp_path_to_{cfg['to_cluster']}.png"))
    
    # samdp.save_txt(f'{base_path}/samdp.txt')

def main(args, cfg: Dict[str, Any]) -> None:
    if not args.ld:
        dataset_path = dataset_creation(cfg['DATASET_CREATION'])
    else:
        data_cfg = cfg['DATASET_CREATION']
        dataset_path = f"/nfs/lslab2/arlin/data_zoo/{data_cfg['environment']}/{data_cfg['algo_str']}-{data_cfg['num_datapoints']}.pkl"
    
    dataset = XRLDataset(dataset_path=dataset_path)
        
    embeddings, clusters = get_data(cfg['GET_DATA'],
                                    dataset,
                                    load_embeddings=args.le,
                                    load_clusters=args.lc)
    
    run_dir = f"./outputs/{cfg['DATASET_CREATION']['environment']}-{cfg['GET_DATA']['CLUSTERS']['num_clusters']}_clusters/"
    
    if not args.sla:
        graph_latent_analytics(run_dir, embeddings, clusters, dataset)
        
    if not args.sca:
        graph_cluster_analytics(run_dir, dataset, clusters)
    
    if not args.ss:
        samdp(run_dir, cfg['SAMDP'], clusters, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ld', help='load dataset', action='store_true')
    parser.add_argument('-le', help='load embeddings', action='store_true')
    parser.add_argument('-lc', help='load clusters', action='store_true')
    parser.add_argument('-sla', help='skip latent analysis', action='store_true')
    parser.add_argument('-sca', help='skip cluster analysis', action='store_true')
    parser.add_argument('-ss', help='skip samdp', action='store_true')
    args = parser.parse_args()
    
    # Logging and warning setup
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    config = get_config('./config.yaml')
    main(args, config)
    