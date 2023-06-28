import numpy as np
import statistics

from arlin.data_analysis.xrl_dataset import XRLDataset
from arlin.data_analysis.analytics_graphing import GraphData, COLORS

class ClusterGrapher():
    
    def __init__(self,
                 dataset: XRLDataset,
                 clusters: np.ndarray):
        
        self.dataset = dataset
        self.clusters = clusters
        self.num_clusters = len(np.unique(self.clusters))
        
        _, cluster_sizes = np.unique(self.clusters, return_counts=True)
        start_cluster_ids, start_cluster_sizes = np.unique(self.clusters[self.dataset.start_indices], return_counts=True)
        done_cluster_ids, done_cluster_sizes = np.unique(self.clusters[self.dataset.done_indices], return_counts=True)
        
        total_starts = sum(start_cluster_sizes)
        total_dones = sum(done_cluster_sizes)
        
        start_clusters = [cluster_id for cluster_id, cluster_size 
                          in zip(start_cluster_ids, start_cluster_sizes) 
                          if (cluster_size / total_starts) > (1 / len(start_cluster_ids))]
        
        done_clusters = [cluster_id for cluster_id, cluster_size 
                         in zip(done_cluster_ids, done_cluster_sizes) 
                         if (cluster_size / total_dones) > (1 / len(start_cluster_ids))]
        
        self.cluster_stage_colors = []
        
        for cluster_id in range(self.num_clusters):
            if cluster_id in start_clusters and cluster_id in done_clusters:
                self.cluster_stage_colors.append('y')
            elif cluster_id in start_clusters:
                self.cluster_stage_colors.append('g')
            elif cluster_id in done_clusters:
                self.cluster_stage_colors.append('r')
            else:
                self.cluster_stage_colors.append('k')

    def cluster_confidence(self) -> GraphData:
        
        cluster_conf = [[] for _ in range(self.num_clusters)]
        
        for e, i in enumerate(self.clusters):
            conf = np.amax(self.dataset.dist_probs[e]).astype(np.float64)
            cluster_conf[i].append(conf)
            
        means = []
        stdevs = []
        
        for i in range(self.num_clusters):
            means.append(statistics.mean(cluster_conf[i]))
            
            try:
                stdevs.append(statistics.stdev(cluster_conf[i]))
            except:
                stdevs.append(0)
        
        title = "Cluster Confidence Analysis"
        
        cluster_conf_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            error_bars=stdevs,
            xlabel='Cluster ID',
            ylabel='Mean Highest Action Confidence',
            showall=True
        )

        return cluster_conf_data

    def cluster_rewards(self) -> GraphData:
        
        cluster_reward = [[] for _ in range(self.num_clusters)]
        
        for e, i in enumerate(self.clusters):
            total_rew = self.dataset.total_rewards[e].astype(np.float64)
            cluster_reward[i].append(total_rew)
            
        means = []
        stdevs = []
        
        for i in range(self.num_clusters):
            means.append(statistics.mean(cluster_reward[i]))
            stdevs.append(statistics.stdev(cluster_reward[i]))
        
        title = "Cluster Reward Analysis"
        
        cluster_reward_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            error_bars=stdevs,
            xlabel='Cluster ID',
            ylabel='Mean Total Reward',
            showall=True
        )
        
        return cluster_reward_data