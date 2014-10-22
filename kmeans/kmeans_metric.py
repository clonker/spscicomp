import numpy as np
from abc import ABCMeta, abstractmethod

class KmeansMetric:
    """ abstract metric """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def dist(self, p, q):
        raise NotImplementedError('subclasses must override dist()!')

class EuclideanMetric(KmeansMetric):
    """ standard euclidean metric """

    def dist(self, p, q):
        return np.linalg.norm(p-q)