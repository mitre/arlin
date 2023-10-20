import os

import numpy as np
import pytest
from matplotlib.patches import Patch

from arlin.analysis.visualization import (
    COLORS,
    GraphData,
    graph_individual_data,
    graph_multiple_data,
)
from arlin.analysis.visualization.visualization import _find_subplot_dims


@pytest.fixture
def graph_data():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([2, 4, 6, 8, 10])
    title = "Test"
    colors = COLORS[0:5]

    handles = [Patch(color=COLORS[i], label=str(i)) for i in range(5)]
    labels = [f"Test {i}" for i in range(5)]
    leg_title = "Test Groups"
    legend = {"handles": handles, "labels": labels, "title": leg_title}

    cmap = "viridis"
    error_bars = [0.5, 0.5, 0.5, 0.5, 0.5]
    xlabel = "Time Steps"
    ylabel = "Values"
    showall = True

    graphdata = GraphData(
        x, y, title, colors, legend, cmap, error_bars, xlabel, ylabel, showall
    )

    return graphdata


class TestVisualization:
    def test_get_data(self, graph_data):
        data = graph_data.get_data()
        assert np.array_equal(data["x"], graph_data.x)
        assert np.array_equal(data["y"], graph_data.y)
        assert data["title"] == graph_data.title
        assert data["legend"] == graph_data.legend
        assert data["cmap"] == graph_data.cmap
        assert data["error_bars"] == graph_data.error_bars
        assert data["xlabel"] == graph_data.xlabel
        assert data["ylabel"] == graph_data.ylabel
        assert data["showall"] == graph_data.showall

        assert len(data.keys()) == 10

    def test_find_subplot_dims(self):
        nrows, ncols = _find_subplot_dims(4, False)
        assert nrows == ncols == 2
        nrows, ncols = _find_subplot_dims(9, False)
        assert nrows == ncols == 3
        nrows, ncols = _find_subplot_dims(3, False)
        assert nrows == 3
        assert ncols == 1
        nrows, ncols = _find_subplot_dims(3, True)
        assert nrows == 1
        assert ncols == 3
        nrows, ncols = _find_subplot_dims(6, False)
        assert nrows == 3
        assert ncols == 2
        nrows, ncols = _find_subplot_dims(6, True)
        assert nrows == 2
        assert ncols == 3

    def test_graph_multiple_data(self, tmpdir, graph_data):
        file_path = os.path.join(tmpdir, "./multiple_graphs.png")
        graph_multiple_data(file_path, "Test", [graph_data, graph_data])
        assert os.path.isfile(file_path)

    def test_graph_individual_data(self, tmpdir, graph_data):
        file_path = os.path.join(tmpdir, "./individual_graphs.png")
        graph_individual_data(file_path, graph_data)
        assert os.path.isfile(file_path)
