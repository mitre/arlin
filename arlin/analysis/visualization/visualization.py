from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from math import isqrt, sqrt
import os
import logging

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
        return dim_long, dim_short

def graph_multiple_data(
    file_path: str,
    figure_title: str,
    graph_datas: List[GraphData],
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
            
            axis.set_title(data.title)
            axis.title.set_size(10)
            
            if not data.showall:
                axis.axis('off')
            else:
                axis.set_xticks(data.x)
                axis.set_xticklabels(axis.get_xticks(), rotation = 90)
                axis.set_xlabel(data.xlabel)
                axis.set_ylabel(data.ylabel)
            
            if data.legend is not None and len(data.legend['labels']) < 4:
                extra_legends = {"bbox_to_anchor": (1.05, 1.0), "loc": 'upper left'}
                data.legend.update(extra_legends)
                axis.legend(**data.legend)
            
            if data.cmap is not None:
                plt.colorbar(scp, ax=axis)
            
            if data.error_bars is not None:
                for i in range(len(data.x)):
                    axis.errorbar(data.x[i], 
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
    filename: str,
    data: GraphData,
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
        plt.xticks(rotation=90)
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