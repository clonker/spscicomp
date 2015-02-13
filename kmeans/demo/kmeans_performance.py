import unittest
import timeit

from spscicomp.kmeans import *
from spscicomp.common.common_data_importer import *
from spscicomp.kmeans.kmeans_data_generator import *
from spscicomp.kmeans.kmeans_plot import *


class TestKmeansTimed(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_kmeans(self):
        print "default_kmeans, average from 10 iterations:"
        print timeit.timeit('kmeans.calculate_centers(3, initial_centers=initial_centers)',
                            setup="import numpy as np;"
                                  "from kmeans import DefaultKmeans;"
                                  "from common.common_data_importer import CommonFileDataImporter;"
                                  "importer = CommonFileDataImporter(filename='test_kmeans_random_data_generator.txt');"
                                  "kmeans = DefaultKmeans(importer=importer);"
                                  "initial_centers = [np.array([11.36545498, 32.76316854]),"
                                  "np.array([44.56166088, 3.98325672]),np.array([3.70092085, 36.24628609])]",
                            number=10)


class TestCextensionTimed(unittest.TestCase):
    def setUp(self):
        data_generator = KmeansRandomDataGenerator(10000, 10, 50)
        data_generator.to_file('test_kmeans_performance.txt')

    def test_default_kmeans(self):
        print "default_kmeans, average from 10 iterations:"
        print timeit.timeit('kmeans.calculate_centers(50)',
                            setup="import numpy as np;"
                                  "from kmeans import DefaultKmeans;"
                                  "from common.common_data_importer import CommonFileDataImporter;"
                                  "importer = CommonFileDataImporter(filename='test_kmeans_performance.txt');"
                                  "kmeans = DefaultKmeans(importer=importer);",
                            number=10)

    def test_default_kmeans_extension(self):
        print "default_kmeans_extension, average from 10 iterations:"
        print timeit.timeit('kmeans.calculate_centers(50)',
                            setup="import numpy as np;"
                                  "from kmeans import DefaultKmeans;"
                                  "from common.common_data_importer import CommonFileDataImporter;"
                                  "importer = CommonFileDataImporter(filename='test_kmeans_performance.txt');"
                                  "kmeans = DefaultKmeans(importer=importer, c_extension=True);",
                            number=10)


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
        importer = CommonFileDataImporter(filename='test_kmeans_random_data_generator.txt')
        kmeans = DefaultKmeans(importer=importer, chunk_size=100)
        initial_centers = [np.array([11.36545498, 32.76316854]),
                           np.array([44.56166088, 3.98325672]),
                           np.array([3.70092085, 36.24628609])]
        history = None
        centers, _ = kmeans.calculate_centers(3, return_centers=True)
        # centers, history = kmeans.calculate_centers(3, initial_centers=initial_centers, save_history=True)
        print "default_kmeans: " + str(centers)
        if history:
            for i in xrange(len(history)):
                plot = KmeansPlot(history[i])
                importer.rewind()
                plot.plot_data(importer.get_data(1000))
                plot.plot_centers()
                plot.save_plot("plots\default_kmeans_%s" % str(i))
        else:
            plot = KmeansPlot(centers)
            importer.rewind()
            plot.plot_data(importer.get_data(1000))
            plot.plot_centers()
            plot.show_plot()


"""
    main
"""

if __name__ == '__main__':
    unittest.main()