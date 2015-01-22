#include "Tica_CExtension.h"

#define PY_INITERROR return NULL
#define TICA_ARRAY_DIMENSION 2
#define TICA_FIRST_ROW 0
#define TICA_FIRST_COL 0

struct sTica_state 
{
    PyObject *error;
};

#define GETSTATE( module ) ( ( struct sTica_state* )PyModule_GetState( module ) )

static int ticaC_traverse( PyObject *module, visitproc visit, void *arg ) 
{
    Py_VISIT( GETSTATE( module )->error );
    return 0;
}

static int ticaC_clear( PyObject *module ) 
{
    Py_CLEAR( GETSTATE( module )->error );
    return 0;
}

PyObject *computeChunkCov( PyArrayObject *i_data )
{
    /* Output matrix:

    | * * * * * |
    | 0 * * * * |
    | 0 0 * * * |
    | 0 0 0 * * |
    | 0 0 0 0 * |
    
    */
    int i, j, k, mCov, nCov, nData, mData;
    int dimCov[2];   
    double *rowCov, *colData1, *colData2;

    ticaC_numpyArrayDim dimData = NULL;
    PyObject *o_cov             = NULL;
    PyArrayObject *cov          = NULL;

    dimData = PyArray_DIMS( i_data );
    dimCov[0] = dimData[1];
    dimCov[1] = dimData[1];
    
    mCov = dimCov[0];
    nCov = dimCov[1];
    nData = dimData[0];
    mData = dimData[1];

    o_cov = PyArray_SimpleNew( TICA_ARRAY_DIMENSION, dimCov, NPY_DOUBLE );
    cov = (PyArrayObject*) PyArray_FROM_OTF( o_cov, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
    
    if ( NULL == o_cov )
    {
        return NULL;
    }

    for ( i = 0; i < mCov; i++ )
    {
        rowCov = (double*) PyArray_GETPTR2( cov, i, i );

        for ( j = i; j < nCov; j++, rowCov++ )
        {
            colData1 = (double*) PyArray_GETPTR2( i_data, TICA_FIRST_ROW, i );
            colData2 = (double*) PyArray_GETPTR2( i_data, TICA_FIRST_ROW, j );

            for ( k = 0; k < nData; k++, colData1 += mData, colData2 += mData )
            {
                *rowCov += (*colData1) * (*colData2);
            }
        }
    }  


    o_cov = (PyObject*)cov;
    Py_INCREF( cov );
    return  o_cov;
}


PyObject *matrixMulti(PyArrayObject *i_data, double tmp)
{
	int i, j, mCov, nCov, nData, mData;
	int dimCov[2];
	double *rowCov;

	ticaC_numpyArrayDim dimData = NULL;
	PyObject *o_cov = NULL;
	PyArrayObject *cov = NULL;
	double *tmp2;

	dimData = PyArray_DIMS(i_data);
	dimCov[0] = dimData[1];
	dimCov[1] = dimData[1];

	mCov = dimCov[0];
	nCov = dimCov[1];
	nData = dimData[0];
	mData = dimData[1];

	o_cov = PyArray_SimpleNew(TICA_ARRAY_DIMENSION, dimCov, NPY_DOUBLE);
	cov = (PyArrayObject*)PyArray_FROM_OTF(o_cov, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

	if (NULL == o_cov)
	{
		return NULL;
	}

	for (i = 0; i < mCov; i++)
	{
		rowCov = (double*)PyArray_GETPTR2(cov, i, i);

		for (j = i; j < nCov; j++, rowCov++)
		{
			tmp2 = (double*)PyArray_GETPTR2(i_data, TICA_FIRST_ROW, i);
			*rowCov += *tmp2 * tmp;

		}
	}

	o_cov = (PyObject*)cov;
	Py_INCREF(cov);
	return  o_cov;
}


PyObject *computeFullMatrix(PyArrayObject *matrix)
{
	/* Output matrix:

	| * * * * * |
	| 0 * * * * |
	| 0 0 * * * |
	| 0 0 0 * * |
	| 0 0 0 0 * |

	*/
	int i, j, mCov, nCov, nData, mData;
	int dimCov[2];
	double *colData1, *colData2;

	ticaC_numpyArrayDim dimData = NULL;
	PyObject *o_cov = NULL;
	PyArrayObject *cov = NULL;

	dimData = PyArray_DIMS(matrix);
	dimCov[0] = dimData[1];
	dimCov[1] = dimData[1];

	mCov = dimCov[0];
	nCov = dimCov[1];
	nData = dimData[0];
	mData = dimData[1];

	o_cov = PyArray_SimpleNew(TICA_ARRAY_DIMENSION, dimCov, NPY_DOUBLE);
	cov = (PyArrayObject*)PyArray_FROM_OTF(o_cov, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

	if (NULL == o_cov)
	{
		return NULL;
	}

	for (i = 0; i < mCov; i++)
	{
		for (j = i; j < nCov; j++)
		{
			colData1 = (double*)PyArray_GETPTR2(matrix, i, j);
			colData2 = (double*)PyArray_GETPTR2(matrix, j, i);
			*colData2 = *colData1;
		}
	}


	o_cov = (PyObject*)cov;
	Py_INCREF(cov);
	return  o_cov;
}


static PyObject *ticaC_computeChunkCov( PyObject *self, PyObject *args ) 
{
    PyObject *inData           = NULL;
    PyArrayObject *inDataArr   = NULL;
    PyObject *o_covMat         = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &inData))
    { 
        return NULL;
    }
    
    inDataArr = (PyArrayObject*) PyArray_FROM_OTF( inData, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );

    o_covMat = computeChunkCov(  inDataArr );
    Py_INCREF( o_covMat );
    
    //return o_covMat;
    return Py_BuildValue( "O", o_covMat);
    
}

PyArrayObject *call_funGetData(PyObject *func, double size) {
	PyObject *args;
	PyObject *kwargs;
	PyObject *result = 0;
	PyArrayObject *retval;

	/* Make sure we own the GIL */
	PyGILState_STATE state = PyGILState_Ensure();

	/* Verify that func is a proper callable */
	if (!PyCallable_Check(func)) {
		fprintf(stderr, "call_func: expected a callable\n");
		goto fail;
	}
	/* Build arguments */
	args = Py_BuildValue("(d)", size);
	kwargs = NULL;

	/* Call the function */
	result = PyObject_Call(func, args, kwargs);
	Py_DECREF(args);
	Py_XDECREF(kwargs);

	/* Check for Python exceptions (if any) */
	if (PyErr_Occurred()) {
		PyErr_Print();
		goto fail;
	}

	/* Verify the result is a float object */
	//if (!PyFloat_Check(result)) {
	//	fprintf(stderr, "call_func: callable didn't return a float\n");
	//	goto fail;
	//}

	/* Create the return value */
	retval = (PyArrayObject *)(result);
	Py_DECREF(result);

	/* Restore previous GIL state and return */
	PyGILState_Release(state);
	return retval;

fail:
	Py_XDECREF(result);
	PyGILState_Release(state);
	return NULL;
	abort();   // Change to something more appropriate
}

bool call_funHasMoreData(PyObject *func) {
	PyObject *args;
	//PyObject *kwargs;
	int result;
	bool retval;

	/* Make sure we own the GIL */
	PyGILState_STATE state = PyGILState_Ensure();

	/* Verify that func is a proper callable */
	if (!PyCallable_Check(func)) {
		fprintf(stderr, "call_func: expected a callable\n");
		goto fail;
	}
	/* Build arguments */
	/* args = Py_BuildValue("") */
	args = NULL;
	//kwargs = NULL;

	/* Call the function */
	//result = PyObject_IsTrue(PyObject_Call(func, args, kwargs));
	result = PyObject_IsTrue(PyObject_CallObject(func, args));
	//Py_DECREF(args);
	//Py_XDECREF(kwargs);

	/* Check for Python exceptions (if any) */
	if (PyErr_Occurred()) {
		PyErr_Print();
		goto fail;
	}

	/* Verify the result is a float object */
	//if (!PyFloat_Check(result)) {
	//	fprintf(stderr, "call_func: callable didn't return a float\n");
	//	goto fail;
	//}

	/* Create the return value */
	//retval = (bool)(result);
	retval = result != 0;
	//Py_DECREF(result);

	/* Restore previous GIL state and return */
	PyGILState_Release(state);
	return retval;

fail:
	Py_XDECREF(result);
	PyGILState_Release(state);
	return NULL;
	abort();   // Change to something more appropriate
}

PyObject *computeCov( PyObject *i_funGetData
                      , PyObject *i_funHasMoreData
                      , double i_chunkSize
					  , double dimData)
{
	PyObject *o_covMat = NULL;
    PyArrayObject *chunk = NULL;
	PyArrayObject *covArr = NULL;
	bool moreData;
	double tmp;

	ticaC_numpyArrayDim dimData2 = NULL;
	PyObject *o_cov = NULL;
	PyArrayObject *cov = NULL;

	chunk = call_funGetData(i_funGetData, i_chunkSize);
	o_covMat = computeChunkCov(chunk);

	moreData = call_funHasMoreData(i_funHasMoreData);

	while (moreData == true)
	{
		chunk = call_funGetData(i_funGetData, i_chunkSize);
		o_covMat =+ computeChunkCov(chunk);
		moreData = call_funHasMoreData(i_funHasMoreData);
	}

	covArr = (PyArrayObject*)PyArray_FROM_OTF(o_covMat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	tmp = 1.0 / (dimData - 1.0);
	o_covMat = matrixMulti(covArr, tmp);
	covArr = (PyArrayObject*)PyArray_FROM_OTF(o_covMat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	o_covMat = computeFullMatrix(covArr);

	//o_covMat = tmp * covArr;

	return (PyObject *)o_covMat;
	//Py_BuildValue("O", o_covMat);
}

static PyObject *ticaC_computeCov( PyObject *self, PyObject *args )
{
    PyObject *funGetData = NULL;
    PyObject *funHasMoreData = NULL;
    double chunkSize;
	double dimData;

    PyArrayObject *inDataArr = NULL;
    PyObject *o_covMat = NULL;

    if (!PyArg_ParseTuple( args, "OOdd", &funGetData, &funHasMoreData, &chunkSize, &dimData))
    {
        return NULL;
    }

    o_covMat = computeCov( funGetData, funHasMoreData, chunkSize, dimData);
    Py_INCREF( o_covMat );

    return o_covMat;
    //return Py_BuildValue( "d", o_covMat);
	//return NULL;
}

static PyMethodDef TicaCMethods[] =
{
    { "computeChunkCov", (PyCFunction)ticaC_computeChunkCov, METH_VARARGS, "Compute the chunk covariance matrix of given data chunk." },
    { "computeCov", (PyCFunction)ticaC_computeCov, METH_VARARGS, "Compute the chunk covariance matrix of given data chunk." },
    { 0, 0, 0, 0 }
};

static struct PyModuleDef TicaCModule = 
{
    PyModuleDef_HEAD_INIT,
    "TicaC",
    NULL,
    sizeof( struct sTica_state ),
    TicaCMethods,
    NULL,
    ticaC_traverse,
    ticaC_clear,
    NULL
};


PyMODINIT_FUNC
PyInit_ticaC( void )
{    
    PyObject *module;
    import_array();

    module = PyModule_Create( &TicaCModule );
    if ( module == NULL )
    {
        PY_INITERROR;
    }

    struct sTica_state *st = GETSTATE( module );

    st->error = PyErr_NewException( "ticaC.Error", NULL, NULL );
    if ( st->error == NULL ) 
    {
        Py_DECREF( module );
        PY_INITERROR;
    }

    return module;

}
