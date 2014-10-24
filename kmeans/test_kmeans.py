import unittest
import numpy as np
from kmeans_metric import EuclideanMetric
from kmeans_data_importer import KmeansFileDataImporter


class TestKmeansMetric(unittest.TestCase):
    def setUp(self):
        pass

    def test_euclidean_metric(self):
        metric = EuclideanMetric()
        self.assertEqual(metric.dist(np.array([0, 0]), np.array([0, 0])), 0, 'for points (0,0) and (0,0): Expected distance is zero!')
        self.assertEqual(metric.dist(np.array([0, 0]), np.array([3, 4])), 5, 'for euclidean metric: d((0,0),(3,4))=5.')
        self.assertEqual(metric.dist(np.array([3, 4]), np.array([0, 0])), 5, 'any metric should be symmetric.')

    def test_kmeans_file_data_importer(self):
        importer = KmeansFileDataImporter(filename="test.txt")
        for i in range(0,5):
            while importer.has_more_data():
                chunk = importer.get_data(20)
                print str(len(chunk)) + ":" + str(chunk)
            importer.rewind()



"""
    main
"""
if __name__ == '__main__':
    unittest.main()