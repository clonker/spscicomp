from distutils.core import setup, Extension

import numpy


setup(
    name='spscicomp project',
    packages=['common', 'hmm', 'kmeans', 'tica'],
    url="https://github.com/florianlitzinger/spscicomp",
    include_dirs=[numpy.get_include()], 
    requires=['numpy'],
    ext_modules=[
        Extension('hmm/lib/c', 
            sources=['hmm/lib/c/extension.c', 'hmm/lib/c/hmm.c', 'hmm/lib/c/hmm32.c']),
        Extension('kmeans/extension/kmeans_c_extension', 
            sources=['kmeans/extension/kmeans_c_extension.cpp', 'kmeans/extension/kmeans_chunk_center.cpp']),
#        Extension('tica/extension/ticaC/ticaC',
#            sources = ['tica/extension/ticaC/Tica_CExtension.cpp']),
    ]
)
