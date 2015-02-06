#include "Tica_CExtension.h"

#define PY_INITERROR return NULL
#define TICA_ARRAY_DIMENSION 2
#define TICA_FIRST_ROW 0
#define TICA_FIRST_COL 0
#define TICA_GET_POINTER2D(obj, i, j, dim) ((void *)(obj + \
                                                    (i)*dim[0] + \
                                                    (j)*dim[1]))

struct sTica_state 
{
    PyObject *error;
};

#define GETSTATE( m ) (&_state)
static sTica_state _state;
#define INITERROR return


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
    npy_intp dimCov[2];   
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
    cov = (PyArrayObject*)PyArray_FROM_OTF( o_cov, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );

    if (NULL == o_cov)
    {
      return NULL;
    }

    for (i = 0; i < mCov; i++)
    {
      rowCov = (double*)PyArray_GETPTR2( cov, i, i );

      for (j = i; j < nCov; j++, rowCov++)
      {
        colData1 = (double*)PyArray_GETPTR2( i_data, TICA_FIRST_ROW, i );
        colData2 = (double*)PyArray_GETPTR2( i_data, TICA_FIRST_ROW, j );

        for (k = 0; k < nData; k++, colData1 += mData, colData2 += mData)
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
	npy_intp dimCov[2];
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
		//rowCov = (double*)PyArray_GETPTR2(cov, i, i);

		for (j = i; j < nCov; j++)
		{
			tmp2 = (double*)PyArray_GETPTR2(i_data, i, j);
			rowCov = (double*)PyArray_GETPTR2(cov, i, j);
			//tmp2 = *rowCov;
			*rowCov = *tmp2 * tmp;

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
	npy_intp dimCov[2];
	double *colData1, *colData2;
	double *rowCov, *rowCov2;

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
			rowCov = (double*)PyArray_GETPTR2(cov, j, i);
			rowCov2 = (double*)PyArray_GETPTR2(cov, i, j);
			colData1 = (double*)PyArray_GETPTR2(matrix, i, j);
			*rowCov = *colData1;
			*rowCov2 = *colData1;
		}
	}

	o_cov = (PyObject*)cov;
	Py_INCREF(cov);
	return  o_cov;
}

PyObject *addMatrixPiecewise(PyObject *matrix, PyObject *matrix2)
{
	int i, j, mCov, nCov, nData, mData;
	npy_intp dimCov[2];
	double *matrixData1, *matrixData2;
	double *outMatrixDat;

	ticaC_numpyArrayDim dimData = NULL;
	PyObject *outMatrix = NULL;
	PyArrayObject *outMatrixArray = NULL, *matrixArray, *matrixArray2;

	matrixArray = (PyArrayObject*)PyArray_FROM_OTF(matrix, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	matrixArray2 = (PyArrayObject*)PyArray_FROM_OTF(matrix2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

	dimData = PyArray_DIMS(matrixArray);
	dimCov[0] = dimData[1];
	dimCov[1] = dimData[1];

	mCov = dimCov[0];
	nCov = dimCov[1];
	nData = dimData[0];
	mData = dimData[1];

	outMatrix = PyArray_SimpleNew(TICA_ARRAY_DIMENSION, dimCov, NPY_DOUBLE);
	outMatrixArray = (PyArrayObject*)PyArray_FROM_OTF(outMatrix, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

	if (NULL == outMatrix)
	{
		return NULL;
	}

	for (i = 0; i < mCov; i++)
	{
		for (j = 0; j < nCov; j++)
		{
			outMatrixDat = (double*)PyArray_GETPTR2(outMatrixArray, i, j);
			matrixData1 = (double*)PyArray_GETPTR2(matrixArray, i, j);
			matrixData2 = (double*)PyArray_GETPTR2(matrixArray, i, j);
			*outMatrixDat = *matrixData1 + *matrixData2;
		}
	}

	outMatrix = (PyObject*)outMatrixArray;
	Py_INCREF(outMatrixArray);
	return  outMatrix;
}


PyArrayObject *call_funGetData(PyObject *func, double size) 
{
	PyObject *args;
	PyObject *kwargs;
	PyObject *result;
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
  if (!PyArray_Check( result )) {
    fprintf( stderr, "call_func: callable didn't return a array\n" );
    goto fail;
  }

  retval = (PyArrayObject*)PyArray_FROM_OTF( result, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );

	/* Create the return value */
	//retval = (PyArrayObject *)(result);
  //col = (double*)PyArray_GETPTR2( retval, TICA_FIRST_ROW, 0);
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
	PyObject *o_covMat = NULL, *chunkCovMat = NULL;
	PyArrayObject *chunk = NULL;
	PyArrayObject *covArr = NULL;
	PyArrayObject *retval = NULL;
	bool moreData;
	double tmp;
	double* test;

	ticaC_numpyArrayDim dimData2 = NULL;
	PyObject *o_cov = NULL;
	PyArrayObject *cov = NULL;

	chunk = call_funGetData(i_funGetData, i_chunkSize);
	o_covMat = computeChunkCov(chunk);

	retval = (PyArrayObject*)PyArray_FROM_OTF(o_covMat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	test = (double*)PyArray_GETPTR2(retval, 0, 0);

	moreData = call_funHasMoreData(i_funHasMoreData);
	
	while (moreData == true)
	{
		chunk = call_funGetData(i_funGetData, i_chunkSize);
		chunkCovMat = computeChunkCov(chunk);
		o_covMat = addMatrixPiecewise(o_covMat, chunkCovMat);
		moreData = call_funHasMoreData(i_funHasMoreData);
	}

	covArr = (PyArrayObject*)PyArray_FROM_OTF(o_covMat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	tmp = 1.0 / (dimData - 1.0);
	o_covMat = matrixMulti(covArr, tmp);
	covArr = (PyArrayObject*)PyArray_FROM_OTF(o_covMat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	o_covMat = computeFullMatrix(covArr);

	//o_covMat = tmp * covArr;

	return o_covMat;
	//Py_BuildValue("O", o_covMat);
}

PyObject *computeColMeans( PyObject *i_funGetData
                         , PyObject *i_funHasMoreData
                         , double *i_chunkSize )
{
  int i, j, nRow, nCol;
  int numRow = 0;
  double *col, *sumCol;
  double factor = 0;
  int whileIter = 0;

  npy_intp colMeansDim[2];

  void *ptrBuffer;

  ticaC_numpyArrayDim dimChunk = NULL;
  PyObject *o_colMeans = NULL;
  PyArrayObject *dataChunk = NULL;
  //PyArrayObject *colMeans = NULL;
  double *colMeans;

  while (call_funHasMoreData( i_funHasMoreData ))
  {
    whileIter++;
    //free(dataChunk);
    dataChunk = call_funGetData( i_funGetData, *i_chunkSize );
    //return (PyObject*)dataChunk;
    dimChunk = PyArray_DIMS( dataChunk );
    numRow += dimChunk[0];
    nRow = dimChunk[0];
    nCol = dimChunk[1];

    if (1 == whileIter)
    {
      colMeansDim[0] = 1;
      colMeansDim[1] = dimChunk[1];

      /*o_colMeans initialized as row vector*/
      
      //colMeans = (PyArrayObject*)PyArray_FROM_OTF( o_colMeans, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );
      colMeans = (double *)malloc( sizeof( double ) * colMeansDim[0] * colMeansDim[1] );
      memset( colMeans, 0, sizeof( double ) * colMeansDim[0] * colMeansDim[1] );

      //for (i = 0; i < nCol; i++)
      //{
      //  sumCol = (double*)PyArray_GETPTR2( colMeans, TICA_FIRST_ROW, i );
      //  *sumCol = 0;
      //}


    }

    for (i = 0; i < nCol; i++)
    {
      col = (double*)PyArray_GETPTR2( dataChunk, TICA_FIRST_ROW, i );
      //sumCol = (double*)PyArray_GETPTR2( colMeans, TICA_FIRST_ROW, i );

      sumCol = (double*)TICA_GET_POINTER2D( colMeans, i, TICA_FIRST_COL, colMeansDim );

      for (j = 0; j < nRow; j++, col += nCol)
      {
        *sumCol += *col;
      }
      //PyArray_free(sumCol);
      //PyArray_free( col );
      //free(col);
      //free(sumCol);
    }
    //PyArray_free( col );
    //PyArray_free( sumCol );
  }

  factor = 1.0 / (double)numRow;

  for (i = 0; i < colMeansDim[1]; i++)
  {
    sumCol = (double*)TICA_GET_POINTER2D( colMeans, i, TICA_FIRST_COL, colMeansDim );
    *sumCol *= factor;
  }

  o_colMeans = PyArray_SimpleNew( TICA_ARRAY_DIMENSION, colMeansDim, NPY_DOUBLE );
  if (NULL == o_colMeans)
  {
    return NULL;
  }
  
  ptrBuffer = PyArray_DATA( (PyArrayObject*)o_colMeans );
  memcpy( ptrBuffer, colMeans, sizeof( double ) * colMeansDim[0] * colMeansDim[1] );

  free(colMeans);

  //o_colMeans = (PyObject*)colMeans;
  Py_INCREF( o_colMeans );
  return  o_colMeans;

}


static PyObject *ticaC_computeChunkCov( PyObject *self, PyObject *args )
{
  PyObject *inData = NULL;
  PyArrayObject *inDataArr = NULL;
  PyObject *o_covMat = NULL;

  if (!PyArg_ParseTuple( args, "O!", &PyArray_Type, &inData ))
  {
    return NULL;
  }

  inDataArr = (PyArrayObject*)PyArray_FROM_OTF( inData, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY );

  o_covMat = computeChunkCov( inDataArr );
  Py_INCREF( o_covMat );

  //return o_covMat;
  return Py_BuildValue( "O", o_covMat );

}

static PyObject *ticaC_computeCov( PyObject *self, PyObject *args )
{
    PyObject *funGetData = NULL;
    PyObject *funHasMoreData = NULL;
    double chunkSize;
	double dimData;
	double *test, *test2, *test3, *test4;

	PyArrayObject *inDataArr = NULL, *covArr;
    PyObject *o_covMat = NULL;

    if (!PyArg_ParseTuple( args, "OOdd", &funGetData, &funHasMoreData, &chunkSize, &dimData))
    {
        return NULL;
    }

    o_covMat = computeCov( funGetData, funHasMoreData, chunkSize, dimData);
    //Py_INCREF( o_covMat );

	covArr = (PyArrayObject*)PyArray_FROM_OTF(o_covMat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	test = (double*)PyArray_GETPTR2(covArr, 0, 0);
	test2 = (double*)PyArray_GETPTR2(covArr, 1, 1);
	test3 = (double*)PyArray_GETPTR2(covArr, 0, 1);
	test4 = (double*)PyArray_GETPTR2(covArr, 1, 0);

    return o_covMat;
    //return Py_BuildValue( "d", o_covMat);
	//return NULL;
}

static PyObject *ticaC_computeColMeans( PyObject *self, PyObject *args )
{
  PyObject *funGetData = NULL;
  PyObject *funHasMoreData = NULL;
  double chunkSize;

  PyArrayObject *inDataArr = NULL;
  PyObject *o_colMeans = NULL;

  if (!PyArg_ParseTuple( args, "OOd", &funGetData, &funHasMoreData, &chunkSize ))
  {
    return NULL;
  }

  o_colMeans = computeColMeans( funGetData, funHasMoreData, &chunkSize );
  //Py_INCREF( o_colMeans );
  //return o_colMeans;
  return Py_BuildValue( "O", o_colMeans );
}


static PyMethodDef TicaCMethods[] =
{
    { "computeChunkCov", (PyCFunction)ticaC_computeChunkCov, METH_VARARGS, "Compute the chunk covariance matrix of given data chunk." },
    { "computeCov", (PyCFunction)ticaC_computeCov, METH_VARARGS, "Compute the chunk covariance matrix of given data chunk." },
    { "computeColMeans", (PyCFunction)ticaC_computeColMeans, METH_VARARGS, "Compute the column means of given data chunk." },
    { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC
//PyInit_ticaC( void )
initticaC(void)
{    
    PyObject *module;
    import_array();
    module = Py_InitModule("ticaC", TicaCMethods );
    
    if ( module == NULL )
    {
       INITERROR;
    }

    struct sTica_state *st = GETSTATE(module);

    st->error = PyErr_NewException( "ticaC.Error", NULL, NULL );
    if ( st->error == NULL ) 
    {
        Py_DECREF( module );
        INITERROR;
    }

    Py_INCREF(st->error);
    PyModule_AddObject( module, "error", st->error );

}
