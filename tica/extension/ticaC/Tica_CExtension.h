#ifndef _TICA_C_EXTENSION_
#define _TICA_C_EXTENSION_

#ifdef _DEBUG
#define _DEBUG_WAS_DEFINED 1
#undef _DEBUG
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"


#ifdef __cplusplus 
extern "C" {
#endif 

    typedef npy_intp *ticaC_numpyArrayDim;

    extern PyObject *computeChunkCov(PyArrayObject *i_data);
    extern PyObject *computeCov( PyObject *i_funGetData
                                ,PyObject *i_funHasMoreData
                                ,PyObject *i_funWriteData
                                ,double *i_chunkSize );



#ifdef __cplusplus  
}
#endif 

#ifdef _DEBUG_WAS_DEINFED
#define _DEBUG 1
#endif

#endif /* !defined(_TICA_C_EXTENSION_) */