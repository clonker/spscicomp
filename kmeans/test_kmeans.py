import unittest
import timeit

from kmeans import *
from kmeans_metric import *
from kmeans_data_importer import *
from kmeans_data_generator import *
from kmeans_plot import *


# TODO: cleanup + split


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

    def test_kmeans_random_data_generator(self):
        data_generator = KmeansRandomDataGenerator(1000, 2, 3)
        data_generator.to_file('test_kmeans_random_data_generator.txt')

    def test_kmeans_file_data_importer(self):
        importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt')
        for i in xrange(0, 5):
            while True:
                chunk = importer.get_data(1000)
                print len(chunk)
                if not importer.has_more_data():
                    break
            importer.rewind()


class TestKmeansPlot(unittest.TestCase):
    def setUp(self):
        pass

    def test_kmeans_plot(self):
        data_generator = KmeansRandomDataGenerator(1000, 2, 3)
        centers = data_generator.get_centers()
        plot = KmeansPlot(centers)
        plot.plot_data(data_generator.get_data())
        plot.plot_centers()
        plot.show_plot()


class TestKmeans(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_kmeans(self):
        importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt')
        kmeans = DefaultKmeans(importer=importer)
        initial_centers = [np.array([11.36545498, 32.76316854]),
                           np.array([44.56166088, 3.98325672]),
                           np.array([3.70092085, 36.24628609])]
        centers, history = kmeans.calculate_centers(3, initial_centers=initial_centers, save_history=True)
        print "default_kmeans: " + str(centers)
        for i in xrange(len(history)):
            plot = KmeansPlot(history[i])
            plot.plot_data(importer.get_data(1000))
            importer.rewind()
            plot.plot_centers()
            plot.save_plot("plots/default_kmeans_%s" % str(i))

    def test_soft_kmeans(self):
        importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt')
        kmeans = SoftKmeans(importer=importer, max_steps=20)
        initial_centers = [np.array([11.36545498, 32.76316854]),
                           np.array([44.56166088, 3.98325672]),
                           np.array([3.70092085, 36.24628609])]
        centers, history = kmeans.calculate_centers(3, initial_centers=initial_centers, save_history=True)
        print "soft_kmeans: " + str(centers)
        for i in xrange(len(history)):
            plot = KmeansPlot(history[i])
            plot.plot_data(importer.get_data(1000))
            importer.rewind()
            plot.plot_centers()
            plot.save_plot("plots/soft_kmeans_%s" % str(i))

    def test_mini_batch_kmeans(self):
        importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt')
        kmeans = MiniBatchKmeans(importer=importer, max_steps=20)
        initial_centers = [np.array([11.36545498, 32.76316854]),
                           np.array([44.56166088, 3.98325672]),
                           np.array([3.70092085, 36.24628609])]
        centers, history = kmeans.calculate_centers(3, initial_centers=initial_centers, save_history=True)
        print "mini_batch_kmeans: " + str(centers)
        for i in xrange(len(history)):
            plot = KmeansPlot(history[i])
            plot.plot_data(importer.get_data(1000))
            importer.rewind()
            plot.plot_centers()
            plot.save_plot("plots/mini_batch_kmeans_%s" % str(i))


class TestKmeansTimed(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_kmeans(self):
        print "default_kmeans, average from 10 iterations:"
        print timeit.timeit('kmeans.calculate_centers(3, initial_centers=initial_centers, save_history=True)',
                            setup="import numpy as np;"
                                  "from kmeans import DefaultKmeans;"
                                  "from kmeans_data_importer import KmeansFileDataImporter;"
                                  "importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt');"
                                  "kmeans = DefaultKmeans(importer=importer);"
                                  "initial_centers = [np.array([11.36545498, 32.76316854]),"
                                  "np.array([44.56166088, 3.98325672]),np.array([3.70092085, 36.24628609])]",
                            number=10)

    def test_soft_kmeans(self):
        print "soft_kmeans, average from 10 iterations:"
        print timeit.timeit('kmeans.calculate_centers(3, initial_centers=initial_centers, save_history=True)',
                            setup="import numpy as np;"
                                  "from kmeans import SoftKmeans;"
                                  "from kmeans_data_importer import KmeansFileDataImporter;"
                                  "importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt');"
                                  "kmeans = SoftKmeans(importer=importer, max_steps=20);"
                                  "initial_centers = [np.array([11.36545498, 32.76316854]),"
                                  "np.array([44.56166088, 3.98325672]),np.array([3.70092085, 36.24628609])]",
                            number=10)

    def test_mini_batch_kmeans(self):
        print "mini_batch_kmeans, average from 10 iterations:"
        print timeit.timeit('kmeans.calculate_centers(3, initial_centers=initial_centers, save_history=True)',
                            setup="import numpy as np;"
                                  "from kmeans import MiniBatchKmeans;"
                                  "from kmeans_data_importer import KmeansFileDataImporter;"
                                  "importer = KmeansFileDataImporter(filename='test_kmeans_random_data_generator.txt');"
                                  "kmeans = MiniBatchKmeans(importer=importer, max_steps=20);"
                                  "initial_centers = [np.array([11.36545498, 32.76316854]),"
                                  "np.array([44.56166088, 3.98325672]),np.array([3.70092085, 36.24628609])]",
                            number=10)


"""
    main
"""
if __name__ == '__main__':
    unittest.main()