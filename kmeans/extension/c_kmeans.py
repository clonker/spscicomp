import kmeans_c_extension as kmc
from kmeans.kmeans import DefaultKmeans
from kmeans_metric import EuclideanMetric


class CKmeans(DefaultKmeans):
    """
    An implementation of the k-means algorithm in C. Refer to the :class:`.DefaultKmeans` class for parameters and
    public methods.
    """

    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100):
        super(CKmeans, self).__init__(metric, importer, chunk_size, max_steps)

    def _iterate(self, centers, centers_list, data):
        data_assigns = [0] * len(data)
        centers_list.append(kmc.cal_chunk_centers(data, centers, data_assigns))
        self._data_assigns.extend(data_assigns)
