from abc import ABCMeta, abstractmethod
from kmeans_metric import EuclideanMetric

class Kmeans:
    """ abstract metric """
    __metaclass__ = ABCMeta

    def __init__(self, metric=EuclideanMetric()):
        self._metric = metric

    @abstractmethod
    def calculate_centers(self, k, data):
        raise NotImplementedError('subclasses must override calculate_centers()!')


class DefaultKmeans(Kmeans):
    def calculate_centers(self, k, data):
        return ''


class WeakKmeans(Kmeans):
    def calculate_centers(self, k, data):
        return ''