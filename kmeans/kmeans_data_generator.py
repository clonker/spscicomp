from abc import ABCMeta, abstractmethod
import numpy as np


class KmeansDataGenerator:
    """ Abstract data generator. Implementations are expected to override the generate_data method. """

    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_data(self):
        raise NotImplementedError('subclasses must override generate_data()!')


class KmeansRandomDataGenerator(KmeansDataGenerator):
    """
    Generate a test dataset for the k-means algorithm. The centers are generated uniformly.
    The other points are produced randomly near one of the centers with normal distribution.

    :param size: Number of data points to generate.
    :type size: int
    :param dimension: Dimension of the euclidean space the data points will belong to.
    :type dimension: int
    :param centers_count: Number of cluster centers around which the data points are to be generated.
    :type centers_count: int
    """
    def __init__(self, size, dimension, centers_count):
        super(KmeansDataGenerator, self).__init__()
        self._size = size
        self._dimension = dimension
        self._centers_count = centers_count
        self._centers = None
        self._data = None
        self.generate_centers()
        self.generate_data()

    def generate_centers(self):
        self._centers = [list(np.random.uniform(0, 100, (1, self._dimension))) for _ in xrange(self._centers_count)]

    def generate_data(self):
        self._data = np.zeros((self._size, self._dimension))
        for i in xrange(self._size):
            center = np.random.randint(0, self._centers_count)
            noise = np.random.normal(0, 5, (1, self._dimension))
            self._data[i, :] = self._centers[center] + noise

    def get_centers(self):
        """
        Return the generated cluster centers.

        :return: A list of numpy arrays representing the cluster centers.
        :rtype: np.array[]
        """
        return self._centers

    def get_data(self):
        """
        Return the generated data points.

        :return: A numpy array of size *size*x*dimension*.
        :rtype: np.array
        """
        return self._data

    def to_file(self, filename):
        """
        Save the generated data to a text file using :func:`numpy.savetxt` which can be read later using the
        respective :class:`.CommonDataImporter` object.

        :param filename: The file name.
        :type filename: str
        """
        np.savetxt(filename, self._data)

    def to_binary_file(self, filename):
        """
        Save the generated data to a binary file using :func:`numpy.save` which can be read later using the
        respective :class:`.CommonDataImporter` object.

        :param filename: The file name.
        :type filename: str
        """
        np.save(filename, self._data)
