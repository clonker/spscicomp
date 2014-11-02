from abc import ABCMeta, abstractmethod
import numpy as np
import Extension.kmeans_C_extension as kmc

from kmeans_metric import EuclideanMetric


class Kmeans:
    """ abstract k-means algorithm """
    __metaclass__ = ABCMeta

    def __init__(self, metric=EuclideanMetric(), importer=None):
        self._metric = metric
        self._importer = importer

    @abstractmethod
    def calculate_centers(self, k):
        raise NotImplementedError('subclasses must override calculate_centers()!')


class DefaultKmeans(Kmeans):
    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100, c_extension=False):
        super(DefaultKmeans, self).__init__(metric, importer)
        self._max_steps = max_steps
        self._chunk_size = chunk_size
        self._dimension = None
        self._c_extension = c_extension

    def calculate_centers(self, k, initial_centers=None, save_history=False):
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
        if save_history:
            return centers, history
        else:
            return centers

    def kmeans_iterate(self, centers):
        centers_list = []
        self._importer.rewind()
        while True:
            data = self._importer.get_data(self._chunk_size)
            if not data:
                break
            else:
                if self._c_extension is False:
                    centers_list.append(self.kmeans_chunk_center(data, centers))
                else:
                    centers_list.append(kmc.cal_chunk_centers(data, centers))
            if not self._importer.has_more_data():
                break
        center_sum = np.zeros([len(centers), len(centers[0])])
        #print centers_list
        for chunk_center in centers_list:
            center_sum += chunk_center
        return center_sum/len(centers_list)

    def kmeans_chunk_center(self, data, centers):
        k = len(centers)
        centers_counter = np.zeros(k)
        new_centers = [np.zeros(self._dimension) for _ in xrange(k)]
        for p in data:
            closest_center = self.closest_center(p, centers)
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


class MiniBatchKmeans(Kmeans):
    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, batch_size=20, max_steps=100):
        super(MiniBatchKmeans, self).__init__(metric, importer)
        self._batch_size = batch_size
        self._max_steps = max_steps
        self._chunk_size = chunk_size
        self._dimension = None

    def calculate_centers(self, k, initial_centers=None, save_history=False):
        data = self._importer.get_data(self._chunk_size)
        self._dimension = data[0].shape[0]
        if initial_centers:
            centers = initial_centers
        else:
            centers = [data[np.random.randint(0, len(data))] for _ in xrange(k)]
        centers_counter = np.zeros(k)
        history = []
        while True:
            for i in xrange(1, self._max_steps):
                if save_history:
                    history.append(centers[:])  # we need to _copy_ the old centers since they get modified in place
                                                # in the iterate method (the iterate methods for default_kmeans
                                                # and soft_kmeans start afresh instead)
                centers, centers_counter = self.mini_batch_kmeans_iterate(data, centers, centers_counter)
            data = self._importer.get_data(self._chunk_size)
            if not self._importer.has_more_data():
                break
        self._importer.rewind()

        if save_history:
            return centers, history
        else:
            return centers

    def mini_batch_kmeans_iterate(self, data, centers, centers_counter):
        mini_batch_centers = [list([]) for _ in xrange(self._batch_size)]
        for i in xrange(0, self._batch_size):
            j = np.random.randint(0, 1000)
            closest_center = self.closest_center(data[j], centers)
            mini_batch_centers[closest_center].append(j)
        for i, mini_batch_center in enumerate(mini_batch_centers):
            for j in mini_batch_center:
                centers_counter[i] += 1
                eta = 1.0 / centers_counter[i]
                centers[i] = centers[i] * (1.0 - eta) + data[j] * eta

        return centers, centers_counter

    def closest_center(self, p, centers):
        min_dist = float('inf')
        closest_center = 0
        for i, center in enumerate(centers):
            dist = self._metric.dist(p, center)
            if dist < min_dist:
                min_dist = dist
                closest_center = i

        return int(closest_center)


class SoftKmeans(DefaultKmeans):
    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100, beta=0.5):
        super(SoftKmeans, self).__init__(metric, importer)
        self._max_steps = max_steps
        self._chunk_size = chunk_size
        self._dimension = None
        self._beta = beta

    def kmeans_chunk_center(self, data, centers):
        k = len(centers)
        data_len = len(data)
        new_centers = [np.zeros(self._dimension) for _ in xrange(k)]
        responsibility = []
        for p in data:
            responsibility.append(self.weight_distance(p, centers))
        sum_respon = np.zeros(k)
        for i, center in enumerate(new_centers):
            for j in xrange(data_len):
                new_centers[i] += responsibility[j][i] * data[j]
                sum_respon[i] += responsibility[j][i]
            new_centers[i] /= sum_respon[i]
        return new_centers

    def weight_distance(self, p, centers):
        k = len(centers)
        distance = np.zeros(k)
        for i, center in enumerate(centers):
            distance[i] = np.exp(-1 * self._beta * self._metric.dist(p, center))
        distance /= np.sum(distance)
        return distance

