__author__ = 'rickwg'

# setup.py
from distutils.core import setup, Extension
import numpy

setup(name='ticaC',
      ext_modules=[
        Extension('ticaC',
					sources = ['Tica_CExtension.cpp'],
				#include_dirs = ['/some/dir'],
                  # define_macros = [('FOO','1')],
                  # undef_macros = ['BAR'],
                 # library_dirs = ['D:\Studium\FU\GitHub\spscicomp_tica\tica\extension\ticaC\Release'],
                  #libraries = ['ticaC.pyd']
                  )
        ],
		include_dirs=[numpy.get_include()]
)