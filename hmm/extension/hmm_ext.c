#include <Python.h>
#include <numpy/arrayobject.h>

#include "hmm.h"

static char forward_doc[]
		= "This function calculates the forward coefficients in the hmm kernel.";

static char backward_doc[]
		= "This function calculates the backward coefficients in the hmm kernel.";

static char compute_gamma_doc[]
		= "This function calculates the backward coefficients in the hmm kernel.";

static char compute_xi_doc[]
		= "This function calculates the backward coefficients in the hmm kernel.";

static char update_model_doc[]
		= "Updates a given model.";

static PyObject *
py_forward(PyObject *self, PyObject *args)
{
	PyArrayObject *pAlpha, *pScale, *pA, *pB, *pPi, *pO;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
			&PyArray_Type, &pAlpha,
			&PyArray_Type, &pScale,
			&PyArray_Type, &pA,
			&PyArray_Type, &pB,
			&PyArray_Type, &pPi,
			&PyArray_Type, &pO)) {
		return NULL;
	}
	int T = PyArray_DIM(pO, 0);
	int N = PyArray_DIM(pA, 0);
	int M = PyArray_DIM(pB, 1);
	const double *A  = (double*)PyArray_DATA(pA);
	const double *B  = (double*)PyArray_DATA(pB);
	const double *pi = (double*)PyArray_DATA(pPi);
	const long *O = (long*)PyArray_DATA(pO);
	double *alpha = (double*)PyArray_DATA(pAlpha);
	double *scale = (double*)PyArray_DATA(pScale);
	double logprob = forward(A, B, pi, O, N, M, T, alpha, scale);
	return Py_BuildValue("d", logprob);
}

static PyObject *
py_backward(PyObject *self, PyObject *args)
{
	PyArrayObject *pA, *pB, *pPi, *pO, *pScale, *pBeta;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
			&PyArray_Type, &pBeta,
			&PyArray_Type, &pScale,
			&PyArray_Type, &pA,
			&PyArray_Type, &pB,
			&PyArray_Type, &pPi,
			&PyArray_Type, &pO)) {
		return NULL;
	}
	int T = PyArray_DIM(pO, 0);
	int N = PyArray_DIM(pA, 0);
	int M = PyArray_DIM(pB, 1);
	const double *A  = (double*)PyArray_DATA(pA);
	const double *B  = (double*)PyArray_DATA(pB);
	const double *pi = (double*)PyArray_DATA(pPi);
	const long *O = (long*)PyArray_DATA(pO);
	double *beta = (double*)PyArray_DATA(pBeta);
	const double *scale = (double*)PyArray_DATA(pScale);
	backward(A, B, pi, O, N, M, T, beta, scale);
	return Py_BuildValue("O", pBeta);
}

static PyObject *
py_gamma(PyObject *self, PyObject *args)
{
	PyArrayObject *pAlpha, *pBeta, *pGamma;
	if (!PyArg_ParseTuple(args, "O!O!O!",
			&PyArray_Type, &pAlpha,
			&PyArray_Type, &pBeta,
			&PyArray_Type, &pGamma)) {
		return NULL;
	}
	int T = PyArray_DIM(pAlpha, 0);
	int N = PyArray_DIM(pAlpha, 1);
	const double *alpha = (double*)PyArray_DATA(pAlpha);
	const double *beta  = (double*)PyArray_DATA(pBeta);
	double *gamma = (double*)PyArray_DATA(pGamma);
	computeGamma(alpha, beta, T, N, gamma);
	return Py_BuildValue("O", pGamma);
}

static PyObject *
py_xi(PyObject *self, PyObject *args)
{
	PyArrayObject *pA, *pB, *pPi, *pO, *pXi, *pAlpha, *pBeta;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
			&PyArray_Type, &pA,
			&PyArray_Type, &pB,
			&PyArray_Type, &pPi,
			&PyArray_Type, &pO,
			&PyArray_Type, &pAlpha,
			&PyArray_Type, &pBeta,
			&PyArray_Type, &pXi))
	{
		return NULL;
	}
	int T = PyArray_DIM(pO, 0);
	int N = PyArray_DIM(pA, 0);
	int M = PyArray_DIM(pB, 1);
	const double *A  = (double*)PyArray_DATA(pA);
	const double *B  = (double*)PyArray_DATA(pB);
	const double *pi = (double*)PyArray_DATA(pPi);
	const long *O = (long*)PyArray_DATA(pO);
	const double *beta = (double*)PyArray_DATA(pBeta);
	const double *alpha = (double*)PyArray_DATA(pAlpha);
	double *xi    = (double*)PyArray_DATA(pXi);
	computeXi(A, B, pi, O, N, M, T, alpha, beta, xi);
	return Py_BuildValue("O", pXi);
}


static PyObject *
py_update(PyObject *self, PyObject *args)
{
	PyArrayObject *pA, *pB, *pPi, *pO, *pXi, *pGamma;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
			&PyArray_Type, &pA,
			&PyArray_Type, &pB,
			&PyArray_Type, &pPi,
			&PyArray_Type, &pO,
			&PyArray_Type, &pGamma,
			&PyArray_Type, &pXi))
	{
		return NULL;
	}
	int T = PyArray_DIM(pO, 0);
	int N = PyArray_DIM(pA, 0);
	int M = PyArray_DIM(pB, 1);
	double *A  = (double*)PyArray_DATA(pA);
	double *B  = (double*)PyArray_DATA(pB);
	double *pi = (double*)PyArray_DATA(pPi);
	const long *O = (long*)PyArray_DATA(pO);
	const double *gamma = (double*)PyArray_DATA(pGamma);
	const double *xi    = (double*)PyArray_DATA(pXi);
	update(A, B, pi, O, N, M, T, gamma, xi);
	return Py_BuildValue("(O,O,O)", pA, pB, pPi);
}

/***
 * structure to describe our methods to python.
 */
static PyMethodDef HmmMethods[] = {
		{"forward", py_forward, METH_VARARGS, forward_doc},
		{"backward", py_backward, METH_VARARGS, backward_doc},
		{"update", py_update, METH_VARARGS, update_model_doc},
		{"compute_gamma", py_gamma, METH_VARARGS, compute_gamma_doc},
		{"compute_xi", py_xi, METH_VARARGS, compute_xi_doc},
		{NULL, NULL, 0, NULL}
};

/***
 * This is the intializer for our functions. Technical stuff
 */
PyMODINIT_FUNC
inithmm_ext(void) {
	(void) Py_InitModule("hmm_ext", HmmMethods);
	import_array();
}

