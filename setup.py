from distutils.core import setup

import numpy


setup(
    name='spscicomp project',
    packages=['common', 'hmm', 'kmeans', 'tica'],
    url="https://github.com/florianlitzinger/spscicomp",
    include_dirs=[numpy.get_include()], requires=['numpy']
)
