from distutils.core import setup, Extension
import numpy

# define the extension module
ext = Extension('c', sources=['extension.c', 'hmm.c', 'hmm32.c'], include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[ext])
