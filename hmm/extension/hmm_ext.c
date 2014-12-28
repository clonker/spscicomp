#include <Python.h>
#include <numpy/arrayobject.h>

#define XI(t,i,j) *(double*)(PyArray_GETPTR3(xi, t, i, j))
#define ALPHA(i,j) *(double*)(PyArray_GETPTR2(alpha, i, j))
#define BETA(i,j) *(double*)(PyArray_GETPTR2(beta, i, j))
#define A(i,j) *(double*)(PyArray_GETPTR2(a, i, j))
#define B(k,i) *(double*)(PyArray_GETPTR2(b, k, i))
#define PI(i) *(double*)(PyArray_GETPTR1(pi, i))
#define O(i) *(int*)(PyArray_GETPTR1(obs,i))
#define SCALING(i) *(double*)(PyArray_GETPTR1(scale,i))
#define GAMMA(t,i) *(double*)(PyArray_GETPTR2(gamma, t, i))

static char forward_doc[]
 = "This function calculates the forward coefficients in the hmm kernel.";

static char backward_doc[]
 = "This function calculates the backward coefficients in the hmm kernel.";

static char update_model_doc[]
 = "Updates a given model.";
/***
 * forward(alpha, scale, A, B, pi, obs)
 * This function calculates the forward coefficients in the hmm kernel.
 *
 * The calculations are done inplace in `alpha'. All input variables are
 * numpy-arrays.
 *
 * - alpha  the place where to do calculations
 * - scale  store scaling factors here
 * - A      is the transition matrix
 * - B      is the state probability matrix
 * - pi     the initial state distribution
 * - obs    the observation array
 *
 * It returns the tuple (alpha, scaling)
 *
 ***/
static PyObject *
forward(PyObject *self, PyObject *args) {
	// get arguments from python
	PyArrayObject *alpha, *scale, *a, *b, *pi, *obs;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
				&PyArray_Type, &alpha,
				&PyArray_Type, &scale,
				&PyArray_Type, &a,
				&PyArray_Type, &b,
				&PyArray_Type, &pi,
				&PyArray_Type, &obs
	)) {
		return NULL;
	}
	// prepare looping
	npy_intp T = PyArray_DIM(alpha, 0);
	npy_intp N = PyArray_DIM(alpha, 1);
	npy_intp i,j,t; // loop indices
	double sum = 0.0;
	// set initial values
	SCALING(0) = 0.0;
	for (i = 0; i < N; i++) {
		ALPHA(0,i) = PI(i)*B(i,O(i));
		SCALING(0) += ALPHA(0,i);
	}
	// set proper scaling now
	for (i = 0; i < N; i++) {
		ALPHA(0,i) /= SCALING(0);
	}

	// the computation. O(T * (N^2 + N))
	for (t = 0; t+1 < T; t++) {
		SCALING(t+1) = 0;
		for (i = 0; i < N; i++) {
			sum = 0.0;
			for (j = 0; j < N; j++)
				sum += ALPHA(t,j)*A(j,i);
			ALPHA(t+1,i) = sum * B(i,O(t+1));
			SCALING(t+1) += ALPHA(t+1,i);
		}
		// set proper scaling now O(N)
		for (i = 0; i < N; i++)
			ALPHA(t+1,i) /= SCALING(t+1);
	}

	PyObject *result = Py_BuildValue("(O,O)", alpha, scale);
	return result;
}

static PyObject *
backward(PyObject *self, PyObject *args) {
	// get arguments from python
	PyArrayObject *beta, *scale, *a, *b, *pi, *obs;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
				&PyArray_Type, &beta,
				&PyArray_Type, &scale,
				&PyArray_Type, &a,
				&PyArray_Type, &b,
				&PyArray_Type, &pi,
				&PyArray_Type, &obs
	)) {
		return NULL;
	}
	// prepare looping
	npy_intp T = PyArray_DIM(beta, 0);
	npy_intp N = PyArray_DIM(beta, 1);
	npy_intp i,j,t; // loop indices
	double sum;
	
	// set initial values
	for (i = 0; i < N; i++) {
		BETA(T-1,i) = 1.0 / SCALING(T-1);
	}

	// the computation. O(T * N^2)
	for (t = T-2; t >= 0; t--) {
		for (i = 0; i < N; i++) {
			sum = 0.0;
			for (j = 0; j < N; j++)
				sum += A(i,j)*B(j,O(t+1))*BETA(t+1,j);
			BETA(t, i) = sum / SCALING(t);
		}
	}

	Py_INCREF(beta);
	return (PyObject*)beta;
}

static PyObject *
update_model(PyObject *self, PyObject *args) {
	// get arguments from python
	PyArrayObject *a, *b, *alpha, *beta, *pi, *obs, *gamma, *xi;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!",
				&PyArray_Type, &a,
				&PyArray_Type, &b,
				&PyArray_Type, &pi,
				&PyArray_Type, &alpha,
				&PyArray_Type, &beta,
				&PyArray_Type, &obs,
				&PyArray_Type, &gamma,
				&PyArray_Type, &xi
	)) {
		return NULL;
	}
	// prepare looping
	npy_intp T = PyArray_DIM(beta, 0);
	npy_intp N = PyArray_DIM(beta, 1);
	npy_intp K = PyArray_DIM(b, 0);
	npy_intp i,j,t,k; // loop indices
	double sum, gamma_sum, xi_sum, numeratorB;

	/* compute xi */
	for (t = 0; t+1 < T; t++) {
		sum = 0.0;
		for (i = 0; i < N; i++)
			for(j = 0; j < N; j++) {
				XI(t,i,j) = ALPHA(t,i)*BETA(t+1,j)*A(i,j)*B(j,O(t+1));
				sum += XI(t,i,j); 
			}
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++)
				XI(t,i,j) /= sum;
	}

	/* compute gamma */
	for (t = 0; t < T; t++) {
		sum = 0.0;
		for (i = 0; i < N; i++) {
			GAMMA(t,i) = ALPHA(t,i)*BETA(t,i);
			sum += GAMMA(t,i);
		}
		for (i = 0; i < N; i++)
			GAMMA(t,i) /= sum;
	}

	// perform update now...
	for (i = 0; i < N; i++)
		PI(i) = GAMMA(0,i);

	for (i = 0; i < N; i++) {
		gamma_sum = 0.0;
		for (t = 0; t+1 < T; t++)
			gamma_sum += GAMMA(t,i);
		for (j = 0; j < N; j++) {
			xi_sum = 0.0;
			for (t = 0; t+1 < T; t++)
				xi_sum += XI(t,i,j);
			A(i,j) = xi_sum / gamma_sum;
		}
		gamma_sum += GAMMA(T-1,i);
		for (k = 0; k < K; ++k) {
			numeratorB = 0.0;
			for (t = 0; t < T; ++t)
				if (O(t) == k)
					numeratorB += GAMMA(t,i);
			B(i,k) = numeratorB / gamma_sum;
		}
	}

	PyObject *result = Py_BuildValue("(O,O,O)", a, b, pi);
	return result;
}


/***
 * structure to describe our methods to python.
 */
static PyMethodDef HmmMethods[] = {
	{"forward", forward, METH_VARARGS, forward_doc},
	{"backward", backward, METH_VARARGS, backward_doc},
	{"update_model", update_model, METH_VARARGS, update_model_doc},
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
