from kmeans import DefaultKmeans


def kmeans(k, importer=None):
    kmeans_implementation = None
    if importer:
        try:
            from cuda.cuda_kmeans import CUDAKmeans

            kmeans_implementation = CUDAKmeans(importer=importer)
        except:
            try:
                from opencl.opencl_kmeans import OpenCLKmeans

                kmeans_implementation = OpenCLKmeans(importer=importer)
            except:
                try:
                    from extension.c_kmeans import CKmeans

                    kmeans_implementation = CKmeans(importer=importer)
                except:
                    print 'Failed to initialize any of the optimized kmeans implementations, using default one'

        if not kmeans_implementation:
            kmeans_implementation = DefaultKmeans(importer=importer)
    else:
        print 'Needs data importer!'
    print 'implementation chosen = ' + str(type(kmeans_implementation))
    if kmeans_implementation and importer:
        c, _ = kmeans_implementation.calculate_centers(k, return_centers=True)
        return c
    return None
