#include <Python.h>
#include <numpy/arrayobject.h>

#define ALPHA(i,j) *(double*)(PyArray_GETPTR2(alpha, i, j))
#define BETA(i,j) *(double*)(PyArray_GETPTR2(beta, i, j))
#define A(i,j) *(double*)(PyArray_GETPTR2(a, i, j))
#define B(k,i) *(double*)(PyArray_GETPTR2(b, k, i))
#define PI(i) *(double*)(PyArray_GETPTR1(pi, i))
#define O(i) *(int*)(PyArray_GETPTR1(obs,i))
#define SCALING(i) *(double*)(PyArray_GETPTR1(scale,i))
#define GAMMA(t,i) (gamma[t*N+i])

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

	// compute xi
	double *xi   = (double*) calloc(N*N,sizeof(double));
	double *xi_t = (double*) malloc(N*N*sizeof(double));
	for (t = 0; t < T-1; t++) {
		// calculate xi to time t
		sum = 0;
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++) {
				xi_t[i*N+j] = ALPHA(t,i)*A(i,j)*B(O(t+1),j)*BETA(t+1,j);
				sum += xi_t[i*N+j];
			}
		// normalize xi to time t and to global sum of xi
		for (i = 0; i < N; i++) 
			for (j = 0; j < N; j++) {
				xi_t[i*N+j] /= sum;
				xi[i*N+j] += xi_t[i*N+j];
			}
	}
	free(xi_t);

	double *gamma = (double*)calloc(T*N, sizeof(double));
	for (t = 0; t < T; t++) {
		sum = 0;
		for (i = 0; i < N; i++) {
			GAMMA(t,i) = ALPHA(t,i)*BETA(t,i);
			sum += GAMMA(t,i);
		}
		for (i = 0; i < N; i++)
			GAMMA(t,i) /= sum;
	}

	for (i = 0; i < N; i++)
		PI(i) = GAMMA(0,i);

	// compute new transition matrix A.
	// a_ij = sum_t=1..T-1 xi_t(i,j) / sum_t=1..T-1 gamma_t(i)  
	for (i = 0; i < N; i++) {
		sum = 0;
		for (j = 0; j < N; j++) {
			A(i,j) = xi[i*N+j];
			sum += xi[i*N+j];
		}
		for (j = 0; j < N; j++) {
			A(i,j) /= sum;
		}
	}
	free(xi);

	// compute new observation probability distribution B
	for (i = 0; i < N; i++) {
		sum = 0;
		for (k = 0; k < K; k++) {
			B(k,i) = 0;
			for (t = 0; t < T; t++) {
				if (O(t) == k)
					B(k,i) += GAMMA(t, i);
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
