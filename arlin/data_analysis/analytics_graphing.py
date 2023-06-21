from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from math import isqrt, sqrt
import os
import logging
import statistics

from arlin.data_analysis.xrl_dataset import XRLDataset

COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
          '#7f7f7f', '#bcbd22', '#17becf', '#101010', '#6f2fff', '#0f8f7f', '#c48c5c',
          '#cf0fcf', '#4b0082')

class GraphData():
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        colors: Optional[List[str]] = None,
        legend: Optional[Dict] = None,
        cmap: Optional[str] = None,
        error_bars: Optional[List[float]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        showall: bool = False
    ):
        self.x = x
        self.y = y
        self.title = title
        self.colors = colors
        self.legend = legend
        self.cmap = cmap
        self.error_bars = error_bars
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.showall = showall
    
    def get_data(self) -> Dict[str, Any]:
        data = {
            "x": self.x,
            "y": self.y,
            "title": self.title,
            "colors": self.colors,
            "legend": self.legend,
            "cmap": self.cmap,
            "error_bars": self.error_bars,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "showall": self.showall
        }
        
        return data
    

class LatentGraphData():
    
    def __init__(self, 
                 embeddings: np.ndarray,
                 dataset: XRLDataset
                 ):
        
        self.embeddings = embeddings
        self.dataset = dataset
        self.num_embeddings = len(self.embeddings)
        self.x = embeddings[:,0]
        self.y = embeddings[:,1]
    
    def set_embeddings(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
    
    def set_dataset(self, dataset: XRLDataset) -> None:
        self.dataset = dataset
    
    def embeddings_graph_data(self) -> GraphData:
        """
        Generate data necessary for creating embedding graphs.
        """
        
        colors = ['#5A5A5A'] * self.num_embeddings
        title = "Embeddings"
        
        embed_data = GraphData(
            x=self.x,
            y=self.y,
            title=title,
            colors=colors
        )
        
        return embed_data
    
    def clusters_graph_data(self, clusters: np.ndarray) -> GraphData:
        """
        Generate data necessary for creating cluster graphs.
        """
        num_clusters = len(np.unique(clusters))
        colors = [COLORS[i] for i in clusters]
        title = f"{num_clusters} Clusters"
        
        handles = [Patch(color=COLORS[i], label=str(i))
                   for i in range(num_clusters)]
        labels = [f"Cluster {i}" for i in range(num_clusters)]
        leg_title = "Cluster Groups"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        cluster_data = GraphData(
            x=self.x,
            y=self.y,
            title=title,
            colors=colors,
            legend=legend
        )
        
        return cluster_data

    def decision_boundary_graph_data(self) -> GraphData:
        """
        Generate data necessary for creating decision boundary graphs.
        """
        colors = [COLORS[i] for i in self.dataset.actions]
        title = "Decision Boundaries for Taken Actions"
        
        num_actions = len(np.unique(self.dataset.actions))
        handles = [Patch(color=COLORS[i], label=str(i))
                   for i in range(num_actions)]
        labels = [f"{i}" for i in range(num_actions)]
        leg_title = "Action Values"
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        decision_boundary_data = GraphData(
            x=self.x,
            y=self.y,
            title=title,
            colors=colors,
            legend=legend
        )
        
        return decision_boundary_data

    def episode_prog_graph_data(self) -> GraphData:
        """
        Generate data necessary for creating episode progression graphs.
        """   
        
        try:
            colors = self.dataset.steps
        except:
            raise ValueError("Current dataset does not include 'steps' attribute.")
        title = "Episode Progression"
        
        episode_prog_data = GraphData(
            x=self.x,
            y=self.y,
            title=title,
            colors=colors,
            cmap="viridis"
        )
        
        return episode_prog_data
    
    def confidence_data(self) -> GraphData:
        """
        Generate data necessary for creating episode progression graphs.
        """
        
        try:
            colors = self.dataset.dist_probs
        except:
            error_str = "Current dataset does not include 'dist_probs' attribute."\
                "Confidence data is only available for PPO datasets."
            raise ValueError(error_str)
        
        colors = np.amax(self.dataset.dist_probs, axis=1)
        title = "Policy Confidence in Greedy Action"
        
        conf_data = GraphData(
            x=self.x,
            y=self.y,
            title=title,
            colors=colors,
            cmap="RdYlGn"
        )
        
        return conf_data
    
    def initial_terminal_state_data(self) -> GraphData:
        """
        Generate data necessary for creating initial/terminal state graphs.
        """

        colors = [COLORS[0] if i else '#F5F5F5' for i in self.dataset.dones]
        for i in self.dataset.start_indices:
            colors[i] = COLORS[1] 
        title = "Initial and Terminal States"
        
        handles = [Patch(color=COLORS[1]), Patch(color=COLORS[0])]
        labels = ["Initial", "Terminal"]
        leg_title = "State Type"
        
        legend = {"handles": handles, "labels": labels, "title": leg_title}
        
        state_data = GraphData(
            x=self.x,
            y=self.y,
            title=title,
            colors=colors,
            legend=legend
        )
    
        return state_data
    
def graph_cluster_analysis(dataset: XRLDataset,
                           clusters: np.ndarray,
                           file_path: str) -> GraphData:
    
    num_clusters = len(np.unique(clusters))
    cluster_conf = [[] for _ in range(num_clusters)]
    
    for e, i in enumerate(clusters):
        conf = np.amax(dataset.dist_probs[e]).astype(np.float64)
        cluster_conf[i].append(conf)
        
    means = []
    stdevs = []
    
    for i in range(num_clusters):
        means.append(statistics.mean(cluster_conf[i]))
        stdevs.append(statistics.stdev(cluster_conf[i]))
    
    title = "Cluster Confidence Analysis"
    
    cluster_conf_data = GraphData(
        x=[i for i in range(num_clusters)],
        y=means,
        title=title,
        colors=COLORS[0:num_clusters],
        error_bars=stdevs,
        xlabel='Cluster ID',
        ylabel='Mean Highest Action Confidence',
        showall=True
    )
    
    graph_individual_data(cluster_conf_data, file_path)
    return cluster_conf_data

def _find_subplot_dims(num_plots: int, horizontal: bool) -> Tuple[int, int]:
    
    # if number is a square number
    if num_plots == isqrt(num_plots) ** 2:
        return sqrt(num_plots), sqrt(num_plots)
    
    if num_plots == 2:
        dim_long = 2
        dim_short = 1
    elif num_plots % 2 == 0:
        dim_long = int(num_plots / 2)
        dim_short = 2
    else:
        dim_long = num_plots
        dim_short = 1
    
    if horizontal:
        return dim_short, dim_long
    else:
        return dim_short, dim_long

def graph_multiple_data(
    figure_title: str,
    graph_datas: List[GraphData],
    file_path: str,
    horizontal: bool = True
):
    num_plots = len(graph_datas)
    nrows, ncols = _find_subplot_dims(num_plots, horizontal)
    
    fig, axs = plt.subplots(int(nrows), int(ncols))
    fig.set_size_inches(6*ncols, 4*nrows)
    fig.suptitle(figure_title)
    
    cur_num = 0
    for irow in range(int(nrows)):
        for icol in range(int(ncols)):
            data = graph_datas[cur_num]
            
            if horizontal and nrows == 1:
                axis = axs[icol]
            elif not horizontal and ncols == 1:
                axis = axs[irow]
            else:
                axis = axs[irow, icol]
            
            scp = axis.scatter(
                data.x, 
                data.y, 
                c=data.colors,
                cmap=data.cmap, 
                s=1)
            
            axis.axis('off')
            axis.set_title(data.title)
            axis.title.set_size(10)
            
            if not data.showall:
                plt.axis('off')
            else:
                plt.xticks(data.x)
                plt.xlabel(data.xlabel)
                plt.ylabel(data.ylabel)
            
            if data.legend is not None and len(data.legend['labels']) < 4:
                extra_legends = {"bbox_to_anchor": (1.05, 1.0), "loc": 'upper left'}
                data.legend.update(extra_legends)
                axis.legend(**data.legend)
            
            if data.cmap is not None:
                plt.colorbar(scp, ax=axis)
            
            if data.error_bars is not None:
                for i in range(len(data.x)):
                    plt.errorbar(data.x[i], 
                                data.y[i], 
                                yerr=data.error_bars[i], 
                                ecolor=data.colors[i], 
                                mec=data.colors[i], 
                                mfc=data.colors[i], 
                                fmt="o", 
                                capsize=5)
            
            plt.tight_layout()
            
            cur_num += 1
    
    logging.info(f"Saving combination graph png to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
def graph_individual_data(
    data: GraphData,
    filename: str
    ):
    """Graph given GraphData to a single plot and save a PNG to the given file.

    Args:
        data (GraphData): Data necessary to graph and individual plot.
        filename (str): Name for the PNG file.
    """
    _ = plt.scatter(
        data.x, 
        data.y, 
        c=data.colors, 
        cmap=data.cmap, 
        s=1)
    
    if not data.showall:
        plt.axis('off')
    else:
        plt.xticks(data.x)
        plt.xlabel(data.xlabel)
        plt.ylabel(data.ylabel)
        
    plt.title(data.title)
    
    if data.legend is not None:
        data.legend.update({"bbox_to_anchor": (1.05, 1.0), "loc": 'upper left'})
        plt.legend(**data.legend)
    
    if data.cmap is not None:
        plt.colorbar()
    
    if data.error_bars is not None:
        for i in range(len(data.x)):
            plt.errorbar(data.x[i], 
                         data.y[i], 
                         yerr=data.error_bars[i], 
                         ecolor=data.colors[i], 
                         mec=data.colors[i], 
                         mfc=data.colors[i], 
                         fmt="o", 
                         capsize=5)
    
    plt.tight_layout()
    
    logging.info(f"Saving individual graph png to {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()