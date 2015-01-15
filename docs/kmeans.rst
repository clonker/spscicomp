The k-means algorithm
=====================

The implementation of the k-means algorithm consists of the following modules:

===========
kmeans_main
===========
.. automodule:: kmeans_main
   :members:

======
kmeans
======
.. automodule:: kmeans
   :members:
   
========
c_kmeans
========
.. automodule:: extension.c_kmeans
   :members:
   

===========
cuda_kmeans
===========
.. py:class:: cuda.cuda_kmeans.CUDAKmeans(metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100)

   An implementation of the k-means algorithm in CUDA. Refer to the :class:`.DefaultKmeans` class for parameters and
   public methods.
    
   
=============
opencl_kmeans
=============
.. py:class:: opencl.opencl_kmeans.OpenCLKmeans(metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100)
    
    An implementation of the k-means algorithm in C. Refer to the :class:`.DefaultKmeans` class for parameters and
    public methods.
    
   
=====================
kmeans_data_generator
=====================
.. automodule:: kmeans_data_generator
   :members:
   
