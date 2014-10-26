import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from kmeans import DefaultKmeans


# Plots 2D data only.
class KmeansPlot:
    def __init__(self, centers=None):
        self._centers = centers
        self._colors = cm.rainbow(np.linspace(0, 1, len(centers)))
        self._kmeans = DefaultKmeans()

    def plot_centers(self):
        for i, center in enumerate(self._centers):
            if type(center) is list:
                center = center[0]
            plt.plot(center[0], center[1], linestyle='None', marker='o', color=self._colors[i, :])

    def plot_data(self, data):
        for i, p in enumerate(data):
            j = self._kmeans.closest_center(p, self._centers)
            plt.plot(p[0], p[1], linestyle='None', marker='.', color=self._colors[j])

    def show_plot(self):
        plt.show()