from kmeans import DefaultKmeans
from common.logger import Logger

LOG = Logger(__name__).get()


def kmeans(k, importer=None):
    """
    Initialize and run the k-means algorithm. If any of the optimized implementations (CUDA, OpenCL, C extension) are
    available, they are selected and initialized automatically in the above order. Then the respective
    :func:`kmeans.Kmeans.calculate_centers` method is called and the output is returned.

    :param k: Number of cluster centers to compute.
    :type k: int
    :param importer: A :class:`.CommonDataImporter` object to be used for importing the numerical data.
    :type importer: :class:`.CommonDataImporter`
    :return: An array of integers :math:`[c(x_i)]` where :math:`x_i` is the i-th data point and
             :math:`c(x_i)` is the index of the cluster center to which :math:`x_i` belongs.
    :rtype: int[]
    """

    kmeans_implementation = None
    if importer:
        try:
            from cuda_kmeans import CUDAKmeans

            kmeans_implementation = CUDAKmeans(importer=importer)
        except:
            try:
                from opencl_kmeans import OpenCLKmeans

                kmeans_implementation = OpenCLKmeans(importer=importer)
            except:
                try:
                    from c_kmeans import CKmeans

                    kmeans_implementation = CKmeans(importer=importer)
                except:
                    LOG.error('Failed to initialize any of the optimized kmeans implementations, using default one')

        if not kmeans_implementation:
            kmeans_implementation = DefaultKmeans(importer=importer)
    else:
        LOG.error('Needs data importer!')
    LOG.debug('implementation chosen = ' + str(type(kmeans_implementation)))
    if kmeans_implementation and importer:
        data_assigns = kmeans_implementation.calculate_centers(k)
        return data_assigns
    return None
