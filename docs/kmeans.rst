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
.. c:function:: static PyObject* cal_chunk_centers(PyObject *dummy, PyObject *args)

   Main function of the C extension.

   :param PyObject* args: Pointer to parameters transported from Python.
   :param PyObject* dummy: Not used here.
   :return: The new chunk centers.
   :rtype: PyObject*
   :raises TypeError: Python Arguments parse error!

.. c:function:: void initkmeans_c_extension()

   Initialize the extension module

.. c:var:: PyMethodDef kmeans_c_extensionMethods

   Variable which stores the maps between functions in C and Python

.. c:function:: int closest_center(PyArrayObject *data,int data_lab, PyArrayObject *centers, int cluster_size, int dimension)

   Given the centers and one point and return which center is nearest to the point.

   :param PyArrayObject* data: One point with related dimension.
   :param int data_lab: Index of the point.
   :param PyArrayObject* centers: Current centers.
   :param int cluster_size: Number of clusters.
   :param int dimension: Dimension of each point and center.
   :return: The index of the nearest center.
   :rtype: int

.. c:function:: PyObject* kmeans_chunk_center(PyArrayObject *data, PyArrayObject *centers, PyObject *data_assigns)

   Record the nearest center of each point and renew the centers.

   :param PyArrayObject* data: Pointer to the point set to be calculated.
   :param PyArrayObject* centers: Current centers.
   :param PyObject* data_assigns: For each point record the index of the nearest center.
   :return: The updated centers.
   :rtype: PyObject*
   :raises ValueError: Parameters are of the wrong sizes.
   :raises MemoryError: RAM allocate error. The imported data chunk may be too large.
   :raises MemoryError: RAM release error.
   :raises MemoryError: Error occurs when creating a new PyArray




===========
cuda_kmeans
===========
.. py:class:: cuda.cuda_kmeans.CUDAKmeans(metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100)

   An implementation of the k-means algorithm in CUDA. Refer to the :class:`.DefaultKmeans` class for parameters and
   public methods.

.. c:function:: static PyObject* cal_chunk_centers(PyObject *dummy, PyObject *args)

   Refer to the :c:func:`cal_chunk_centers` in c_kmeans.

.. c:function:: void initkmeans_c_extension_cuda()

   Refer to the :c:func:`initkmeans_c_extension` in c_kmeans.

.. c:var:: PyMethodDef kmeans_c_extensionMethods

   Refer to the :c:data:`kmeans_c_extension_cudaMethods` in c_kmeans.

.. c:function:: _global__ void chunk_centers_sum_cuda(double *cu_data,double *cu_centers, int* cu_centers_counter, double* cu_new_centers,int* cu_data_assigns, int* cluster_size,int *dimension,int *chunk_size)

   Divide the whole data set into several parts, each part is calculated by a Block in cuda.
   After calculating the index of the nearest center, select a thread to add up the related centers in one Block.

   :param double* cu_data: A chunk of points, which are given pointwise.
   :param double* cu_centers: Current centers.
   :param int* cu_centers_counter: Count how many points are nearest to a given center, count blockwise.
   :param double* cu_new_centers: Calculate the sum of the points which are nearest to a given center, add blockwise.
   :param int* cu_data_assigns: The index of the center which is nearest to a given point.
   :param int* cluster_size: Number of clusters
   :param int* dimension: Dimension of the points.
   :param int* chunk_size: Number of points in the chunk.
   :return chunk_centers_sum_cuda: Summation of nearest centers in one block.
   :rtype: double*

.. c:function:: PyObject* kmeans_chunk_center_cuda(PyArrayObject *data, PyArrayObject *centers, PyObject *data_assigns)

   Record the nearest center of each point and renew the centers.

   :param PyArrayObject* data: Pointer to the point set to be calculated.
   :param PyArrayObject* centers: Current centers.
   :param PyObject* data_assigns: For each point record the index of the nearest center.
   :return: The updated centers.
   :rtype: PyObject*
   :raises Exception: No available device detected.
   :raises Exception: Compute compacity of the graphic card is not enough.
   :raises Exception: Only 1 device is supported currently.
   :raises ValueError: Parameters are of the wrong sizes.
   :raises MemoryError: RAM allocate Error. The imported data chunk may be too large.
   :raises MemoryError: RAM release error.
   :raises MemoryError: Graphic card RAM allocate error.
   :raises MemoryError: Graphic card RAM release error.
   :raises MemoryError: Error occurs when creating a new PyArray

   
=============
opencl_kmeans
=============
.. py:class:: opencl.opencl_kmeans.OpenCLKmeans(metric=EuclideanMetric(), importer=None, chunk_size=1000, max_steps=100)
    
    An implementation of the k-means algorithm in OpenCL. Refer to the :class:`.DefaultKmeans` class for parameters and
    public methods.
    
   
=====================
kmeans_data_generator
=====================
.. automodule:: kmeans_data_generator
   :members:
   
