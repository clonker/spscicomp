import numpy as np
from abc import ABCMeta, abstractmethod
from kmeans_metric import EuclideanMetric


class Kmeans:
    """Abstract k-means algorithm. Implementations are expected to override the calculate_centers method."""
    __metaclass__ = ABCMeta

    def __init__(self, metric=EuclideanMetric(), importer=None):
        self._metric = metric
        self._importer = importer

    @abstractmethod
    def calculate_centers(self, k):
        raise NotImplementedError('subclasses must override calculate_centers()!')


class DefaultKmeans(Kmeans):
    """
    Default implementation of the k-means algorithm. Once supplied with an :class:`.CommonDataImporter` object, use the
    calculate_centers method to compute k cluster centers.

    :param metric: A :class:`.KmeansMetric` object to be used for calculating distances between points. The default is
                   the :class:`.EuclideanMetric`.
    :type metric: :class:`.KmeansMetric`
    :param importer: A :class:`.CommonDataImporter` object to be used for importing the numerical data.
    :type importer: :class:`.CommonDataImporter`
    :param chunk_size: The number of data points to be imported and processed at a time.
    :type chunk_size: int
    :param max_steps: The maximum number of steps to run the algorithm for. If the iteration did not converge after
                      this number of steps, the algorithm is terminated and the last result returned.
    :type max_steps: int
    """

    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100):
        super(DefaultKmeans, self).__init__(metric, importer)
        self._max_steps = max_steps
        self._chunk_size = chunk_size
        self._dimension = None
        self._data_assigns = []

    #@profile
    def calculate_centers(self, k, initial_centers=None, return_centers=False, save_history=False):
        """
        Main method of the k-means algorithm. Computes k cluster centers from the data supplied by a
        :class:`.CommonDataImporter` object.

        :param k: Number of cluster centers to compute.
        :type k: int
        :param initial_centers: Array of cluster centers to start the iteration with. If omitted, random data points
                                from the first chunk of data are used.
        :type initial_centers: numpy.array
        :param return_centers: If set to True then the cluster centers are returned.
        :type return_centers: bool
        :param save_history: If this and return_centers is set to True then the cluster centers in each iteration step
                             are returned.
        :type save_history: bool
        :return: An array of integers :math:`[c(x_i)]` where :math:`x_i` is the i-th data point and
                 :math:`c(x_i)` is the index of the cluster center to which :math:`x_i` belongs.
        :rtype: int[]
        :return: An array of the computed cluster centers.
        :rtype: np.array
        :return: A list of arrays of the cluster centers in each iteration step.
        :rtype: np.array[]
        """

        self._importer.rewind()
        data = self._importer.get_data(self._chunk_size)
        self._dimension = data[0].shape[0]
        if initial_centers is not None:
            centers = np.asarray(initial_centers, dtype=np.float32)
        else:
            centers = np.asarray([data[np.random.randint(0, len(data))] for _ in xrange(k)], dtype=np.float32)
        history = []
        self._importer.rewind()
        for i in xrange(1, self._max_steps):
            old_centers = centers
            if save_history:
                history.append(centers)
            centers = self.kmeans_iterate(centers)
            if np.allclose(centers, old_centers, rtol=1e-5):
                break
        if return_centers:
            if save_history:
                return centers, self._data_assigns, history
            else:
                return centers, self._data_assigns
        else:
            return self._data_assigns

    # @profile
    def _iterate(self, centers, centers_list, data):
        """
        Override method for some special implementations, e.g., the c extension.

        :param centers: The current list of centers
        :type centers: numpy.array

        :param centers_list: A list that subsequently contains the currently computed centers, as in every chunk of the
        data file which gets processed, the corresponding centers are being calculated and in the end need
        to be averaged.
        :type centers_list: np.array[]

        :param data: The current chunk of data.
        :type data: np.array

        :return: None
        """
        centers_list.append(self.kmeans_chunk_center(data, centers))

    def kmeans_iterate(self, centers):
        centers_list = []
        self._data_assigns = []  # reset the list once per iteration
        while True:
            data = self._importer.get_data(self._chunk_size)
            if len(data) is 0:
                break
            else:
                self._iterate(centers, centers_list, data)
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
