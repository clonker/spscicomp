#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include <stdio.h>
#include <stdlib.h>

#include "Python.h"
#include "numpy/arrayobject.h"

/* Needs to be compiled as C files because of the Naming problem in Namespace */
#ifdef __cplusplus
extern "C" {
#endif

PyObject* kmeans_chunk_center_cuda(PyArrayObject *data, PyArrayObject *centers, PyObject *data_assigns);

static PyObject* cal_chunk_centers(PyObject *dummy, PyObject *args)
{
    /* Receive the array of data points and the current centers from C,
       calculate the new center and return them as well as the assignment data. */
    PyArrayObject *data = NULL;
    PyObject *centers = NULL;
    PyObject *data_assigns = NULL;
    if (!PyArg_ParseTuple(args, "O!OO!", &PyArray_Type, &data, &centers, &PyList_Type, &data_assigns)){
        PyErr_SetString(PyExc_TypeError, "Python Arguments parse error!");
        return NULL;
    }

    PyObject *chunk_centers = NULL;
    //printf("CUDA_Module called!");
    chunk_centers = kmeans_chunk_center_cuda(data, (PyArrayObject*)centers, data_assigns);
    if (chunk_centers == NULL){
        return NULL;
    }
    Py_INCREF(chunk_centers);  /* The returned list should still exist after calling the C extension */
    return chunk_centers;
}

static PyMethodDef kmeans_c_extension_cudaMethods[] =
{
    /* Mapping between functions in C and Python */
    {"cal_chunk_centers", cal_chunk_centers, METH_VARARGS, "Calculate the centers of one chunk"},
    {NULL, NULL}
};

void initkmeans_c_extension_cuda()
{
    /* Initialize the extension module */
    import_array();
    PyObject* m;
    m = Py_InitModule("kmeans_c_extension_cuda", kmeans_c_extension_cudaMethods);
}

#ifdef __cplusplus
}
#endif
