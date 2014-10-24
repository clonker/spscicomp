from abc import ABCMeta, abstractmethod
import numpy as np


class KmeansDataGenerator:
    """ abstract data generator """

    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_data(self):
        raise NotImplementedError('subclasses must override generate_data()!')


class KmeansRandomDataGenerator(KmeansDataGenerator):
    def __init__(self, size, dimension, centers_count):
        super(KmeansDataGenerator, self).__init__()
        self._size = size
        self._dimension = dimension
        self._centers_count = centers_count
        self._data = None
        self.generate_data()

    def generate_data(self):
        self._data = np.zeros((self._size, self._dimension))
        centers = [list(np.random.uniform(0, 100, (1, self._dimension))) for _ in xrange(self._centers_count)]
        for i in xrange(self._size):
            center = np.random.randint(0, self._centers_count)
            noise = np.random.normal(0, 5, (1, self._dimension))
            self._data[i, :] = centers[center] + noise

    def to_file(self, filename):
        np.savetxt(filename, self._data)