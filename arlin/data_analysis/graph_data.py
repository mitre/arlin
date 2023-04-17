from typing import Any, Dict, List, Optional

import numpy as np

class GraphData():
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str,
        colors: Optional[List[str]],
        legend: Optional[Dict] = None,
        cmap: Optional[str] = None,
        showaxis: bool = False
    ):
        self.x = x
        self.y = y
        self.title = title
        self.colors = colors
        self.legend = legend
        self.cmap = cmap
        self.showaxis = showaxis
    
    def get_data(self) -> Dict[str, Any]:
        data = {
            "x": self.x,
            "y": self.y,
            "title": self.title,
            "colors": self.colors,
            "legend": self.legend,
            "cmap": self.cmap,
            "showaxis": self.showaxis
        }
        
        return data