from abc import ABCMeta, abstractmethod
from numpy import random as rand
import numpy as np

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
    def __init__(self, metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100):
        super(DefaultKmeans, self).__init__(metric, importer)
        self._max_steps = max_steps
        self._chunk_size = chunk_size

    def calculate_centers(self, k):
        data = self._importer.get_data(self._chunk_size)
        d = data[0].shape[0]
        centers = [list(np.zeros(d)) for _ in xrange(k)]

        while True:
            for i in xrange(1, self._max_steps):
                old_centers = centers
                centers = self.kmeans_iterate(data, centers)
        
                if np.array_equal(centers, old_centers):
                    break

            data = self._importer.get_data(self._chunk_size)

            if not self._importer.has_more_data():
                break

        self._importer.rewind()
    
        return centers

    def kmeans_iterate(self, data, centers):
        k = len(centers)
        centers_counter = np.zeros(k)
        new_centers = np.zeros(centers.shape[0], k)
       
        for p in data:
            closest_center = self.closest_center(p, centers)
            new_centers[closest_center] += p
            centers_counter[closest_center] += 1

        for i, center in enumerate(new_centers):
            if centers_counter[i] > 0:
                new_centers[i] /= centers_counter[i]

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

    def calculate_centers(self, k):
        data = self._importer.get_data(self._chunk_size)
        d = data[0].shape[0]
        centers = [list(np.zeros(d)) for _ in xrange(k)]
        centers_counter = np.zeros(k)

        while True:
            for i in xrange(1, self._max_steps):
                centers, centers_counter = self.mini_batch_kmeans_iterate(data, centers, centers_counter)

            data = self._importer.get_data(self._chunk_size)

            if not self._importer.has_more_data():
                break

        self._importer.rewind()

        return centers

    def mini_batch_kmeans_iterate(self, data, centers, centers_counter):
        mini_batch_centers = [list([]) for _ in xrange(self._batch_size)]

        for i in xrange(0, self._batch_size):
            j = rand.random.randint(0, 1000)
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

