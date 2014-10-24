import unittest
import numpy as np

from kmeans_metric import EuclideanMetric
from kmeans_data_importer import KmeansFileDataImporter
from kmeans_data_generator import KmeansRandomDataGenerator
from kmeans_plot import KmeansPlot


class TestKmeans(unittest.TestCase):
    def setUp(self):
        pass

    def test_euclidean_metric(self):
        metric = EuclideanMetric()
        self.assertEqual(metric.dist(np.array([0, 0]), np.array([0, 0])), 0,
                         'for points (0,0) and (0,0): Expected distance is zero!')
        self.assertEqual(metric.dist(np.array([0, 0]), np.array([3, 4])), 5, 'for euclidean metric: d((0,0),(3,4))=5.')
        self.assertEqual(metric.dist(np.array([3, 4]), np.array([0, 0])), 5, 'any metric should be symmetric.')

    def test_kmeans_random_data_generator(self):
        data_generator = KmeansRandomDataGenerator(1000, 2, 3)
        data_generator.to_file('test_kmeans_random_data_generator.txt')

    def test_kmeans_file_data_importer(self):
        importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt')
        for i in range(0, 5):
            while True:
                chunk = importer.get_data(1000)
                print len(chunk)
                if not importer.has_more_data():
                    break
            importer.rewind()

    def test_kmeans_plot(self):
        centers = [list(np.random.uniform(0, 100, (1, 2))) for _ in xrange(3)]
        plot = KmeansPlot(centers)
        plot.show_plot()


"""
    main
"""
if __name__ == '__main__':
    unittest.main()