from distutils.core import setup

import numpy


setup(
    name='kmeans',
    packages=['cuda', 'extension', 'opencl'],
    url="https://github.com/florianlitzinger/spscicomp",
    include_dirs=[numpy.get_include()], requires=['numpy']
)
