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


class TestKmeans(unittest.TestCase):

    def test_kmeans_equilibrium_state(self):
        initial_centers_equilibrium = [np.array([0, 0, 0])]
        importer_equilibrium = KmeansSimpleDataImporter(np.array([
            np.array([1, 1, 1], dtype=np.float), np.array([1, 1, -1], dtype=np.float),
            np.array([1, -1, -1], dtype=np.float), np.array([-1, -1, -1], dtype=np.float),
            np.array([-1, 1, 1], dtype=np.float), np.array([-1, -1, 1], dtype=np.float),
            np.array([-1, 1, -1], dtype=np.float), np.array([1, -1, 1], dtype=np.float)
        ]))
        for use_c_extension in [True, False]:
            kmeans = DefaultKmeans(importer=importer_equilibrium, c_extension=use_c_extension)
            res, _ = kmeans.calculate_centers(
                k=1,
                initial_centers=initial_centers_equilibrium,
                return_centers=True
            )
            self.assertEqual(1, len(res), 'If k=1, there should be only one output center.')
            msg = 'In an equilibrium state the resulting centers should not be different from the initial centers.'
            self.assertTrue(np.array_equal(initial_centers_equilibrium[0], res[0]), msg)

    def test_kmeans_contraction_property(self):
        target = np.array([1000, -1000, 1000], dtype=float)
        for use_c_extension in [True, False]:
            kmeans = DefaultKmeans(
                c_extension=use_c_extension,
                importer=KmeansSimpleDataImporter(np.array([target]))
            )
            res, data, history = kmeans.calculate_centers(
                k=1,
                return_centers=True,
                save_history=True,
                initial_centers=np.array([np.array([0, 0, 0])])
            )
            metric = EuclideanMetric()

            previous_dist = None
            for histEntry in history:
                if previous_dist is not None:
                    self.assertLess(metric.dist(histEntry, target), previous_dist)
                previous_dist = metric.dist(histEntry, target)


"""
    main
"""

if __name__ == '__main__':
    unittest.main()