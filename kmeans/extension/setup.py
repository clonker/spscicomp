from distutils.core import setup, Extension

setup(name='kmeans_c_extension',
      ext_modules=[Extension('kmeans_c_extension',
                             sources=['kmeans_c_extension.cpp', 'kmeans_chunk_center.cpp'])])
