#!/bin/sh

python2 setup.py build_ext --inplace
rm -Rf ./build
#echo $(python2 -c 'from distutils.util import get_platform; print get_platform()')
