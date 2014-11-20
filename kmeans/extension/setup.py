from distutils.core import setup, Extension
import numpy

setup(
    name='kmeans_c_extension',
    ext_modules=[
        Extension('kmeans_c_extension', sources=['kmeans_c_extension.cpp', 'kmeans_chunk_center.cpp'])
    ],
    include_dirs=[numpy.get_include()]
)
