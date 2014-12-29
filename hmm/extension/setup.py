from distutils.core import setup, Extension
import numpy

# define the extension module
hmm_ext = Extension('hmm_ext', sources=['hmm_ext.c', 'hmm.c'], include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[hmm_ext])
