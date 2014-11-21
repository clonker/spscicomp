#!/bin/sh

rm -R build
python2 setup.py build
cp build/lib.linux-x86_64-2.7/kmeans_c_extension.so .
