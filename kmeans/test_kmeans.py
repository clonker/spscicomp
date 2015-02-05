import unittest

from common.common_data_importer import *
from kmeans_data_generator import *
from extension.c_kmeans import *
from os import remove
from opencl.opencl_kmeans import OpenCLKmeans


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
        importer = CommonFileDataImporter(filename=f_name)
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
        importer_equilibrium = CommonSimpleDataImporter(np.array([
            np.array([1, 1, 1], dtype=np.float), np.array([1, 1, -1], dtype=np.float),
            np.array([1, -1, -1], dtype=np.float), np.array([-1, -1, -1], dtype=np.float),
            np.array([-1, 1, 1], dtype=np.float), np.array([-1, -1, 1], dtype=np.float),
            np.array([-1, 1, -1], dtype=np.float), np.array([1, -1, 1], dtype=np.float)
        ]))
        for kmeans in [
            CKmeans(importer=importer_equilibrium),
            DefaultKmeans(importer=importer_equilibrium),
            OpenCLKmeans(importer=importer_equilibrium)
        ]:
            res, _ = kmeans.calculate_centers(
                k=1,
                initial_centers=initial_centers_equilibrium,
                return_centers=True
            )
            self.assertEqual(1, len(res), 'If k=1, there should be only one output center.')
            msg = 'Type=' + str(type(kmeans)) + '. ' + \
                  'In an equilibrium state the resulting centers should not be different from the initial centers.'
            self.assertTrue(np.array_equal(initial_centers_equilibrium[0], res[0]), msg)

    def test_kmeans_contraction_property(self):
        target = np.array([1000, -1000, 1000], dtype=float)
        for kmeans in [
            CKmeans(importer=CommonSimpleDataImporter(np.array([target]))),
            DefaultKmeans(importer=CommonSimpleDataImporter(np.array([target]))),
            OpenCLKmeans(importer=CommonSimpleDataImporter(np.array([target])))
        ]:
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

    def test_kmeans_convex_hull(self):
        points = [
            [-212129 / 100000, -20411 / 50000, 2887 / 5000],
            [-212129 / 100000, 40827 / 100000, -5773 / 10000],
            [-141419 / 100000, -5103 / 3125, 2887 / 5000],
            [-141419 / 100000, 1 / 50000, -433 / 250],
            [-70709 / 50000, 3 / 100000, 17321 / 10000],
            [-70709 / 50000, 163301 / 100000, -5773 / 10000],
            [-70709 / 100000, -204121 / 100000, -5773 / 10000],
            [-70709 / 100000, -15309 / 12500, -433 / 250],
            [-17677 / 25000, -122471 / 100000, 17321 / 10000],
            [-70707 / 100000, 122477 / 100000, 17321 / 10000],
            [-70707 / 100000, 102063 / 50000, 2887 / 5000],
            [-17677 / 25000, 30619 / 25000, -433 / 250],
            [8839 / 12500, -15309 / 12500, -433 / 250],
            [35357 / 50000, 102063 / 50000, 2887 / 5000],
            [8839 / 12500, -204121 / 100000, -5773 / 10000],
            [70713 / 100000, -122471 / 100000, 17321 / 10000],
            [70713 / 100000, 30619 / 25000, -433 / 250],
            [35357 / 50000, 122477 / 100000, 17321 / 10000],
            [106067 / 50000, -20411 / 50000, 2887 / 5000],
            [141423 / 100000, -5103 / 3125, 2887 / 5000],
            [141423 / 100000, 1 / 50000, -433 / 250],
            [8839 / 6250, 3 / 100000, 17321 / 10000],
            [8839 / 6250, 163301 / 100000, -5773 / 10000],
            [106067 / 50000, 40827 / 100000, -5773 / 10000],
        ]
        importer_permutahedron = CommonSimpleDataImporter(np.asarray(points, dtype=float))
        for kmeans in [
            DefaultKmeans(importer=importer_permutahedron),
            CKmeans(importer=importer_permutahedron),
            OpenCLKmeans(importer=importer_permutahedron)
        ]:
            res, data = kmeans.calculate_centers(
                k=1,
                return_centers=True
            )

            # Check hyperplane inequalities. If they are all fulfilled, the center lies within the convex hull.
            self.assertGreaterEqual(np.inner(np.array([-11785060650000, -6804069750000, -4811167325000], dtype=float),
                                             res) + 25000531219381, 0)
            self.assertGreaterEqual(
                np.inner(np.array([-1767759097500, 1020624896250, 721685304875], dtype=float), res) + 3749956484003, 0)
            self.assertGreaterEqual(np.inner(np.array([-70710363900000, -40824418500000, 57734973820000], dtype=float),
                                             res) + 199998509082907, 0)
            self.assertGreaterEqual(np.inner(np.array([70710363900000, 40824418500000, -57734973820000], dtype=float),
                                             res) + 199998705841169, 0)
            self.assertGreaterEqual(np.inner(np.array([70710363900000, -40824995850000, -28867412195000], dtype=float),
                                             res) + 149999651832937, 0)
            self.assertGreaterEqual(np.inner(np.array([-35355181950000, 20412497925000, -28867282787500], dtype=float),
                                             res) + 100001120662259, 0)
            self.assertGreaterEqual(
                np.inner(np.array([23570121300000, 13608139500000, 9622334650000], dtype=float), res) + 49998241292257,
                0)
            self.assertGreaterEqual(np.inner(np.array([0, 577350000, -204125000], dtype=float), res) + 1060651231, 0)
            self.assertGreaterEqual(np.inner(np.array([35355181950000, -20412497925000, 28867282787500], dtype=float),
                                             res) + 99997486799779, 0)
            self.assertGreaterEqual(np.inner(np.array([0, 72168750, 51030625], dtype=float), res) + 176771554, 0)
            self.assertGreaterEqual(np.inner(np.array([0, -288675000, 102062500], dtype=float), res) + 530329843, 0)
            self.assertGreaterEqual(np.inner(np.array([0, 0, 250], dtype=float), res) + 433, 0)
            self.assertGreaterEqual(np.inner(np.array([0, -144337500, -102061250], dtype=float), res) + 353560531, 0)
            self.assertGreaterEqual(np.inner(np.array([0, 0, -10000], dtype=float), res) + 17321, 0)


"""
    main
"""

if __name__ == '__main__':
    unittest.main()