import unittest

from kmeans_metric import *
from kmeans_data_importer import *
from kmeans_data_generator import *
from kmeans_plot import *
from os import remove


class TestKmeansMetric(unittest.TestCase):
    def setUp(self):
        pass

    def test_euclidean_metric(self):
        metric = EuclideanMetric()
        self.assertEqual(metric.dist(np.array([0, 0]), np.array([0, 0])), 0,
                         'for points (0,0) and (0,0): Expected distance is zero!')
        self.assertEqual(metric.dist(np.array([0, 0]), np.array([3, 4])), 5, 'for euclidean metric: d((0,0),(3,4))=5.')
        self.assertEqual(metric.dist(np.array([3, 4]), np.array([0, 0])), 5, 'any metric should be symmetric.')


class TestKmeansData(unittest.TestCase):
    def setUp(self):
        pass

    def test_kmeans_data_generator(self):
        dim = 2
        centers = 4
        n_points = 12345
        data_generator = KmeansRandomDataGenerator(n_points, dim, centers)
        self.assertEqual(centers, len(data_generator.get_centers()), 'Number of generated centers was wrong')
        self.assertEqual(dim, data_generator.get_data()[0].shape[0], 'Dimension of data was wrong')
        self.assertEqual(n_points, len(data_generator.get_data()), 'Number of generated data points was wrong')

    def test_kmeans_file_data_importer(self):
        f_name = 'unittest_data.txt'
        data_generator = KmeansRandomDataGenerator(1234, 2, 3)
        data_generator.to_file(f_name)
        importer = KmeansFileDataImporter(filename=f_name)
        line_count = sum(1 for _ in open(f_name))
        for i in xrange(5):
            curr_line_count = 0
            while True:
                chunk = importer.get_data(1000)
                curr_line_count += len(chunk)
                if not importer.has_more_data():
                    break
            importer.rewind()
            msg = 'Reading file. Expected line count: ' + str(line_count) + ', actual line count: ' + \
                  str(curr_line_count)
            self.assertEqual(line_count, curr_line_count, msg)
        remove(f_name)


"""
    main
"""

if __name__ == '__main__':
    unittest.main()