import numpy as np
import statistics
from matplotlib.patches import Patch

from arlin.dataset import XRLDataset
from arlin.analysis.visualization import GraphData

class ClusterAnalyzer():
    
    def __init__(self,
                 dataset: XRLDataset,
                 clusters: np.ndarray):
        
        self.dataset = dataset
        self.clusters = clusters
        self.num_clusters = len(np.unique(self.clusters))
        
        start_clusters = set(self.clusters[self.dataset.start_indices])
        term_clusters = set(self.clusters[self.dataset.term_indices])
        
        self.cluster_stage_colors = []
        
        for cluster_id in range(self.num_clusters):
            if cluster_id in start_clusters and cluster_id in term_clusters:
                self.cluster_stage_colors.append('y')
            elif cluster_id in start_clusters:
                self.cluster_stage_colors.append('g')
            elif cluster_id in term_clusters:
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
        
        handles = [Patch(color='g'),
                   Patch(color='r'),
                   Patch(color='k')]
        labels = ['Initial', 'Terminal', 'Intermediate']
        leg_title = "Cluster Stage"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        cluster_conf_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            legend=legend,
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
            
            try:
                stdevs.append(statistics.stdev(cluster_reward[i]))
            except:
                stdevs.append(0)
        
        title = "Cluster Reward Analysis"
        
        handles = [Patch(color='g'),
                   Patch(color='r'),
                   Patch(color='k')]
        labels = ['Initial', 'Terminal', 'Intermediate']
        leg_title = "Cluster Stage"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        cluster_reward_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            legend=legend,
            error_bars=stdevs,
            xlabel='Cluster ID',
            ylabel='Mean Total Reward',
            showall=True
        )
        
        return cluster_reward_data
    
    def cluster_values(self) -> GraphData:
        
        cluster_value = [[] for _ in range(self.num_clusters)]
        
        for e, i in enumerate(self.clusters):
            value = self.dataset.critic_values[e].astype(np.float64)
            cluster_value[i].append(value)
            
        means = []
        stdevs = []
        
        for i in range(self.num_clusters):
            means.append(statistics.mean(cluster_value[i]))
            
            try:
                stdevs.append(statistics.stdev(cluster_value[i]))
            except:
                stdevs.append(0)
        
        title = "Cluster Value Analysis"
        
        handles = [Patch(color='g'),
                   Patch(color='r'),
                   Patch(color='k')]
        labels = ['Initial', 'Terminal', 'Intermediate']
        leg_title = "Cluster Stage"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        cluster_reward_data = GraphData(
            x=[i for i in range(self.num_clusters)],
            y=means,
            title=title,
            colors=self.cluster_stage_colors,
            legend=legend,
            error_bars=stdevs,
            xlabel='Cluster ID',
            ylabel='Mean Value',
            showall=True
        )
        
        return cluster_reward_data