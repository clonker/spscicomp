#include <Python.h>
#include <numpy/arrayobject.h>

#define ALPHA(i,j) *(double*)(PyArray_GETPTR2(alpha, i, j))
#define BETA(i,j) *(double*)(PyArray_GETPTR2(beta, i, j))
#define A(i,j) *(double*)(PyArray_GETPTR2(a, i, j))
#define B(k,i) *(double*)(PyArray_GETPTR2(b, k, i))
#define PI(i) *(double*)(PyArray_GETPTR1(pi, i))
#define O(i) *(int*)(PyArray_GETPTR1(obs,i))
#define SCALING(i) *(double*)(PyArray_GETPTR1(scale,i))

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

	// set initial values
	double scaling = 0;
	for (i = 0; i < N; i++) {
		ALPHA(0,i) = PI(i)*B(O(0),i);
		scaling   += ALPHA(0,i);
	}
	// set proper scaling now
	SCALING(0) = 1 / scaling;
	for (i = 0; i < N; i++) {
		ALPHA(0,i) *= SCALING(0);
	}

	// the computation. O(T * (N^2 + N))
	for (t = 1; t < T; t++) {
		scaling = 0;
		// do the recursion O(N^2)
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++)
				ALPHA(t,i) += ALPHA(t-1,j)*A(j,i);
			ALPHA(t,i) *= B(O(t),i);
			scaling += ALPHA(t,i);
		}
		// set proper scaling now O(N)
		SCALING(t) = 1 / scaling;
		for (i = 0; i < N; i++) {
			ALPHA(t,i) *= SCALING(t);
		}
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

	// set initial values
	for (i = 0; i < N; i++) {
		BETA(T-1,i) = SCALING(T-1);
	}

	// the computation. O(T * N^2)
	for (t = T-2; t >= 0; t--) {
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++)
				BETA(t,i) += A(i,j)*B(O(t+1),j)*BETA(t+1,j)*SCALING(t);
		}
	}

	Py_INCREF(beta);
	return (PyObject*)beta;
}

static PyObject *
update_model(PyObject *self, PyObject *args) {
	// get arguments from python
	PyArrayObject *a, *b, *alpha, *beta, *pi, *obs;
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
				&PyArray_Type, &a,
				&PyArray_Type, &b,
				&PyArray_Type, &pi,
				&PyArray_Type, &alpha,
				&PyArray_Type, &beta,
				&PyArray_Type, &obs
	)) {
		return NULL;
	}
	// prepare looping
	npy_intp T = PyArray_DIM(beta, 0);
	npy_intp N = PyArray_DIM(beta, 1);
	npy_intp K = PyArray_DIM(b, 0);
	npy_intp i,j,t,k; // loop indices
	double sum = 0;

	// new initial distribution
	for (i = 0; i < N; i++) {
		PI(i) = ALPHA(0,i)*BETA(0,i);
		sum  += PI(i);
	}
	// normalize pi
	for (i = 0; i < N; i++)
		PI(i) /= sum;

	// compute new transition matrix A
	for (i = 0; i < N; i++) {
		sum = 0;
		for (j = 0; j < N; j++) {
			double aij = A(i,j);
			A(i,j) = 0;
			for (t = 0; t < T-1; t++)
				A(i,j) += ALPHA(t,i)*aij*B(O(t+1),j)*BETA(t+1,j);
			sum += A(i,j);
		}
		// normalize each row
		for (j = 0; j < N; j++)
			A(i,j) /= sum;
	}

	double *gamma = (double*)calloc(T, sizeof(double));
	for (t = 0; t < T; t++)
		for (i = 0; i < N; i++)
			gamma[t] += ALPHA(t,i)*BETA(t,i);

	// compute new observation probability distribution B
	sum = 0;
	for (i = 0; i < N; i++) {
		sum = 0;
		for (k = 0; k < K; k++) {
			B(k,i) = 0;
			for (t = 0; t < T; t++) {
				if (O(t) == k)
					B(k,i) += ALPHA(t,i)*BETA(t,i) / gamma[t];
			}
			sum += B(k,i);
		}
		for (k = 0; k < K; k++)
			B(k,i) /= sum;
	}
	free(gamma);
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
