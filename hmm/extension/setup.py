from distutils.core import setup, Extension
import numpy

# define the extension module
cos_module = Extension('hmm_ext', sources=['hmm.c'],
		include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[cos_module])
