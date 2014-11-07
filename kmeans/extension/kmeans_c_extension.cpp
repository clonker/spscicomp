#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include <stdio.h>
#include <stdlib.h>

#include "Python.h"
#include "numpy/arrayobject.h"

#ifdef __cplusplus 
extern "C" {  
#endif 
  
PyObject* kmeans_chunk_center(PyObject *data, PyArrayObject *centers);

static PyObject* cal_chunk_centers (PyObject *dummy, PyObject *args)
{
    PyObject *data = NULL;
    PyObject *centers = NULL;
    if (!PyArg_ParseTuple(args, "O!O", &PyList_Type, &data, &centers))
        return NULL;
    PyObject *chunk_centers = NULL;
    chunk_centers = kmeans_chunk_center(data, (PyArrayObject*)centers);
    Py_INCREF(chunk_centers);
    return chunk_centers;
}

static PyMethodDef kmeans_C_extensionMethods[] = 
{
    {"cal_chunk_centers", cal_chunk_centers, METH_VARARGS, "Calculate the centers of one chunk"},
    {NULL, NULL}
};

void initkmeans_C_extension() 
{  
    import_array();
    PyObject* m;
    m = Py_InitModule("kmeans_C_extension", kmeans_C_extensionMethods);
}

#ifdef __cplusplus  
}
#endif
