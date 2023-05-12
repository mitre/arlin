import pickle
from typing import Any, List, Tuple
import os
import logging

import matplotlib.pyplot as plt

from math import isqrt, sqrt

from arlin.data_analysis.graph_data import GraphData


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

def find_subplot_dims(num_plots: int, horizontal: bool) -> Tuple[int, int]:
    
    # if number is a square number
    if num_plots == isqrt(num_plots) ** 2:
        return sqrt(num_plots), sqrt(num_plots)
    
    if num_plots % 2 == 0:
        dim_long = int(num_plots / 2)
        dim_short = 2
    else:
        dim_long = num_plots
        dim_short = 1
    
    if horizontal:
        return dim_short, dim_long
    else:
        return dim_short, dim_long

def graph_subplots(
    figure_title: str,
    graph_datas: List[GraphData],
    trial_path: str, 
    filename: str,
    horizontal: bool = True
):
    num_plots = len(graph_datas)
    nrows, ncols = find_subplot_dims(num_plots, horizontal)
    
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
            
            if not data.showaxis:
                plt.axis('off')
            
            if data.legend is not None:
                extra_legends = {"bbox_to_anchor": (1.05, 1.0), "loc": 'upper left'}
                data.legend.update(extra_legends)
                axis.legend(**data.legend)
            
            if data.cmap is not None:
                plt.colorbar(scp, ax=axis)
            
            plt.tight_layout()
            
            cur_num += 1
    
    save_path = os.path.join(trial_path, "combo_graphs", filename)
    logging.info(f"Saving combination graph png to {save_path}...")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()