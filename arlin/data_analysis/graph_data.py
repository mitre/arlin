from typing import Any, Dict, List, Optional

import numpy as np

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
        showaxis: bool = False
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
        self.showaxis = showaxis
    
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
            "showaxis": self.showaxis
        }
        
        return data