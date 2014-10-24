from abc import ABCMeta, abstractmethod
from numpy import random as rand
import numpy as np

from kmeans_metric import EuclideanMetric


class Kmeans:
    """ abstract metric """
    __metaclass__ = ABCMeta

    def __init__(self, metric=EuclideanMetric(), importer=None):
        self._metric = metric
        self._chunk = None
        self._importer = importer

    @abstractmethod
    def calculate_centers(self, k):
        raise NotImplementedError('subclasses must override calculate_centers()!')


class DefaultKmeans(Kmeans):
    def calculate_centers(self, k):
        return ''


class WeakKmeans(Kmeans):
    def calculate_centers(self, k):
        return ''


class MiniBatchKMeans(Kmeans):
    def __init__(self, metric=EuclideanMetric(), importer=None, batch_size=20, max_steps=100):
        super(MiniBatchKMeans, self).__init__(metric, importer)
        self._batch_size = batch_size
        self._max_steps = max_steps

    def miniBatchKmeansIterate(self, data, centers, centers_counter):
        mini_batch_centers = [list([]) for _ in xrange(self._batch_size)]

        for i in xrange(0, self._batch_size):
            j = rand.random.randint(0, 1000)
            closest_center = self.ClosestCenter(data[j], centers)
            mini_batch_centers[closest_center].append(j)

        for i, mini_batch_center in enumerate(mini_batch_centers):
            for j in mini_batch_center:
                centers_counter[i] += 1
                eta = 1.0 / centers_counter[i]
                centers[i] = centers[i] * (1.0 - eta) + data[j] * eta

        return centers, centers_counter

    def calculate_centers(self, k, data):
        centers = None
        centers_counter = np.zeros(k)
        i = 1

        while True:  # not np.array_equal(centers, old_centers):
            centers, centers_counter = self.miniBatchKmeansIterate(data, centers, centers_counter)

            i += 1
            if i > self._max_steps:
                break

        print i

        return centers

    def ClosestCenter(self, p, centers):
        min_dist = float('inf')
        closest_center = 0
        for i, center in enumerate(centers):
            dist = self._metric.dist(p, center)
            if dist < min_dist:
                min_dist = dist
                closest_center = i

        return int(closest_center)

    def zeroCenters(self, k, d):
        centers = []
        for i in xrange(0, k):
            centers.append(np.zeros(d))

        return centers