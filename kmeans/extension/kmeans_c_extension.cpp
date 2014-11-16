#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API

#include <stdio.h>
#include <stdlib.h>

#include "Python.h"
#include "numpy/arrayobject.h"

/*Need to be complied as C files because of the Naming problem in Namespace*/
#ifdef __cplusplus 
extern "C" {  
#endif 
  
PyObject* kmeans_chunk_center(PyObject *data, PyArrayObject *centers, PyObject *data_assigns);

static PyObject* cal_chunk_centers(PyObject *dummy, PyObject *args)
{
    /*Receive the List and Array data from C, calculate the new center and return.*/
    PyObject *data = NULL;
    PyObject *centers = NULL;
    PyObject *data_assigns = NULL;
    if (!PyArg_ParseTuple(args, "O!OO!", &PyList_Type, &data, &centers, &PyList_Type, &data_assigns))
        return NULL;
    PyObject *chunk_centers = NULL;
    chunk_centers = kmeans_chunk_center(data, (PyArrayObject*)centers, data_assigns);
    Py_INCREF(chunk_centers);  /*The turned list should still exist after calling the C extension*/
    return chunk_centers;
}

static PyMethodDef kmeans_c_extensionMethods[] =
{
    /* Mapping between functions in C and Python*/
    {"cal_chunk_centers", cal_chunk_centers, METH_VARARGS, "Calculate the centers of one chunk"},
    {NULL, NULL}
};

void initkmeans_c_extension()
{  
    /*Initial the extension module*/
    import_array();
    PyObject* m;
    m = Py_InitModule("kmeans_c_extension", kmeans_c_extensionMethods);
}

#ifdef __cplusplus  
}
#endif
