from abc import ABCMeta, abstractmethod
import numpy as np
from extension import kmeans_c_extension as kmc

from kmeans_metric import EuclideanMetric


class Kmeans:
    """ Abstract k-means algorithm """
    __metaclass__ = ABCMeta

    def __init__(self, metric=EuclideanMetric(), importer=None):
        self._metric = metric
        self._importer = importer

    @abstractmethod
    def calculate_centers(self, k):
        raise NotImplementedError('subclasses must override calculate_centers()!')


class DefaultKmeans(Kmeans):
    """
    Default implementation of the k-means algorithm. Once supplied with an KmeansDataImporter object, use the
    calculate_centers method to compute k cluster centers.
    """

    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100, c_extension=False):
        super(DefaultKmeans, self).__init__(metric, importer)
        self._max_steps = max_steps
        self._chunk_size = chunk_size
        self._dimension = None
        self._c_extension = c_extension
        self._data_assigns = []

    # @profile
    def calculate_centers(self, k, initial_centers=None, return_centers=False, save_history=False):
        """
        Main method of the k-means algorithm. Computes k cluster centers from the data supplied by a
        KmeansDataImporter object.
        :param k: Number of cluster centers to compute.
        :param initial_centers: Array of cluster centers to start the iteration with. If omitted, random data points
        from the first chunk of data are used.
        :param return_centers: If set to True then the cluster centers are returned.
        :param save_history: If this and return_centers is set to True then the cluster centers in each iteration step
        are returned.
        :return: centers - an array of the computed cluster centers. data_assigns - an array of integers [c(xi)] where
        xi is the i-th data point and c(xi) is the index of the cluster center to which xi belongs. history - a list of
        arrays of the cluster centers in each iteration step.
        """

        self._importer.rewind()
        data = self._importer.get_data(self._chunk_size)
        self._dimension = data[0].shape[0]
        if initial_centers:
            centers = initial_centers
        else:
            centers = [data[np.random.randint(0, len(data))] for _ in xrange(k)]
            centers = np.asarray(centers)
        history = []
        self._importer.rewind()
        for i in xrange(1, self._max_steps):
            old_centers = centers
            if save_history:
                history.append(centers)
            centers = self.kmeans_iterate(centers)
            if np.array_equal(centers, old_centers):
                break
        if return_centers:
            if save_history:
                return centers, self._data_assigns, history
            else:
                return centers, self._data_assigns
        else:
            return self._data_assigns

    # @profile
    def kmeans_iterate(self, centers):
        centers_list = []
        self._data_assigns = []  # reset the list once per iteration
        while True:
            data = self._importer.get_data(self._chunk_size)
            if len(data) is 0:
                break
            else:
                if self._c_extension is False:
                    centers_list.append(self.kmeans_chunk_center(data, centers))
                else:
                    data_assigns = [0] * len(data)
                    centers_list.append(kmc.cal_chunk_centers(data, centers, data_assigns))
                    self._data_assigns.extend(data_assigns)
            if not self._importer.has_more_data():
                break
        self._importer.rewind()
        center_sum = np.zeros([len(centers), len(centers[0])])
        for chunk_center in centers_list:
            center_sum += chunk_center
        return center_sum/len(centers_list)

    def kmeans_chunk_center(self, data, centers):
        k = len(centers)
        centers_counter = np.zeros(k)
        new_centers = [np.zeros(self._dimension) for _ in xrange(k)]
        for p in data:
            closest_center = self.closest_center(p, centers)
            self._data_assigns.append(closest_center)
            new_centers[closest_center] += p
            centers_counter[closest_center] += 1
        for i, center in enumerate(new_centers):
            if centers_counter[i] > 0:
                new_centers[i] /= centers_counter[i]
            else:
                new_centers[i] = centers[i]
        return new_centers

    def closest_center(self, p, centers):
        min_dist = float('inf')
        closest_center = 0
        for i, center in enumerate(centers):
            dist = self._metric.dist(p, center)
            if dist < min_dist:
                min_dist = dist
                closest_center = i

        return int(closest_center)
