import ctypes
import platform
from kmeans import DefaultKmeans

def kmeans(k, importer=None):
    kmeans_implementation = None
    if importer:
        try:
            if platform.system() == "Microsoft":
                _libcudart = ctypes.windll.LoadLibrary('cudart.dll')
            elif platform.system() == "Darwin":
                _libcudart = ctypes.cdll.LoadLibrary('libcudart.dylib')
            else:
                _libcudart = ctypes.cdll.LoadLibrary('libcudart.so')
        except:
            _libcudart = None
        if _libcudart:
            from cuda.cuda_kmeans import CUDAKmeans
            kmeans_implementation = CUDAKmeans(k, importer)
        else:
            try:
                from opencl.opencl_kmeans import OpenCLKmeans
                kmeans_implementation = OpenCLKmeans(k, importer)
            except:
                try:
                    from extension.c_kmeans import CKmeans
                    kmeans_implementation = CKmeans(k, importer)
                except:
                    print 'Failed to initialize any of the optimized kmeans implementations, using default one'

        if not kmeans_implementation:
            kmeans_implementation = DefaultKmeans(k, importer)
    else:
        print 'Needs data importer!'
    print 'implementation chosen = '+str(type(kmeans_implementation))
    if kmeans_implementation and importer:
        return kmeans_implementation.calculate_centers(k)
