#include <Python.h>
#include <numpy/arrayobject.h>

#include "hmm.h"

static char forward_doc[]
=
"Compute P(ob|A,B,pi) and all forward coefficients and scaling coefficients.\n" \
"\n" \
"    Parameters\n" \
"    ----------\n" \
"    A : numpy.array of numpy.float64 and shape (N,N)\n" \
"        transition matrix of the hidden states\n" \
"    B : numpy.array of numpy.float64 and shape (N,M)\n" \
"        symbol probability matrix for each hidden state\n" \
"    pi : numpy.array of numpy.float64 and shape (N)\n" \
"         initial distribution\n" \
"    ob : numpy.array of numpy.int32 and shape (T)\n" \
"         observation sequence of integer between 0 and M, used as indices in B\n" \
"\n" \
"    Returns\n" \
"    -------\n" \
"    prob : numpy.float64\n" \
"           The probability to observe the sequence 'ob' with the model given\n" \
"           by 'A', 'B' and 'pi'.\n" \
"    alpha : numpy.array of numpy.float64 and shape (T,N)\n" \
"            alpha[t,i] is the ith forward coefficient of time t. These can be\n" \
"            used in many different algorithms related to HMMs.\n" \
"    scaling : numpy.array of numpy.float64 and shape (T)\n" \
"\n" \
"    See Also\n" \
"    --------\n" \
"    hmm.kernel.simple.forward : A simple python implementation of this function.\n" \
"    hmm.lib.c.forward32 : single precision variation using 32bit floats.\n";
static PyObject *
py_forward(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pPi, *pO;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
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
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp alpha_dims[2] = {T, N};
    npy_intp scale_dims[1] = {T};
    PyObject *pAlpha = PyArray_ZEROS(2, alpha_dims, NPY_FLOAT64, 0);
    PyObject *pScale = PyArray_ZEROS(1, scale_dims, NPY_FLOAT64, 0);
    
    double *alpha = (double*) PyArray_DATA(pAlpha);
    double *scale = (double*) PyArray_DATA(pScale);
    double logprob = forward(alpha, scale, A, B, pi, O, N, M, T);
    
    PyObject *tuple = Py_BuildValue("(d,O,O)", logprob, pAlpha, pScale);
    Py_DECREF(pAlpha);
    Py_DECREF(pScale);
    return tuple;
}

static PyObject *
py_forward_no_scaling(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pPi, *pO;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
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
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp alpha_dims[2] = {T, N};
    PyObject *pAlpha = PyArray_ZEROS(2, alpha_dims, NPY_FLOAT64, 0);
    
    double *alpha = (double*) PyArray_DATA(pAlpha);
    double logprob = forward_no_scaling(alpha, A, B, pi, O, N, M, T);
    
    PyObject *tuple = Py_BuildValue("(d,O)", logprob, pAlpha);
    Py_DECREF(pAlpha);
    return tuple;
}

static char backward_doc[]
        = "This function calculates the backward coefficients in the hmm kernel.";

static PyObject *
py_backward_no_scaling(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO;
    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO)) {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const double *A  = (double*)PyArray_DATA(pA);
    const double *B  = (double*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp dims[2] = {T, N};
    PyObject *pBeta = PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    
    double *beta = (double*) PyArray_DATA(pBeta);
    backward_no_scaling(beta, A, B, O, N, M, T);

    return pBeta;
}

static PyObject *
py_backward(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO, *pScale;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO,
            &PyArray_Type, &pScale)) {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const double *A  = (double*)PyArray_DATA(pA);
    const double *B  = (double*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    const double *scale = (double*)PyArray_DATA(pScale);
    
    npy_intp dims[2] = {T, N};
    PyObject *pBeta = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    
    double *beta = (double*) PyArray_DATA(pBeta);
    backward(beta, A, B, O, scale, N, M, T);

    return pBeta;
}

static char compute_gamma_doc[]
        = "This function calculates the backward coefficients in the hmm kernel.";

static char compute_xi_doc[]
        = "This function calculates the backward coefficients in the hmm kernel.";

static char update_model_doc[]
        = "Updates a given model.";

static PyObject *
py_gamma(PyObject *self, PyObject *args)
{
    PyArrayObject *pAlpha, *pBeta;
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &pAlpha,
            &PyArray_Type, &pBeta)) {
        return NULL;
    }
    int T = PyArray_DIM(pAlpha, 0);
    int N = PyArray_DIM(pAlpha, 1);
    const double *alpha = (double*)PyArray_DATA(pAlpha);
    const double *beta  = (double*)PyArray_DATA(pBeta);
    
    npy_intp dims[2] = {T, N};
    PyObject *pGamma = PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);

    double *gamma = (double*) PyArray_DATA(pGamma);
    computeGamma(gamma, alpha, beta, T, N);

    return pGamma;
}

static PyObject *
py_nomA(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO, *pAlpha, *pBeta;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
            &PyArray_Type, &pAlpha,
            &PyArray_Type, &pBeta,
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO))
    {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const double *A  = (double*)PyArray_DATA(pA);
    const double *B  = (double*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    const double *beta = (double*)PyArray_DATA(pBeta);
    const double *alpha = (double*)PyArray_DATA(pAlpha);
    
    npy_intp nomA_dims[2] = {N,N};
    PyObject *pNomA = PyArray_ZEROS(2, nomA_dims, NPY_DOUBLE, 0);

    double *nomA = (double*) PyArray_DATA(pNomA);
    compute_nomA(nomA, A, B, O, alpha, beta, N, M, T);
    
    return pNomA;
}

static PyObject *
py_denomA(PyObject *self, PyObject *args)
{
    PyArrayObject *pGamma;
    int T;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &pGamma, &T))
    {
        return NULL;
    }
    int N = PyArray_DIM(pGamma, 1);
    const double *gamma = (double*)PyArray_DATA(pGamma);
    
    npy_intp denomA_dims[1] = {N};
    PyObject *pDenomA = PyArray_ZEROS(1, denomA_dims, NPY_DOUBLE, 0);

    double *denomA = (double*) PyArray_DATA(pDenomA);
    compute_denomA(denomA, gamma, T, N);

    return pDenomA;
}

static PyObject *
py_nomB(PyObject *self, PyObject *args)
{
    PyArrayObject *pO, *pGamma;
    int M;
    if (!PyArg_ParseTuple(args, "O!O!i",
            &PyArray_Type, &pO,
            &PyArray_Type, &pGamma,
            &M))
    {
        return NULL;
    }
    int T = PyArray_DIM(pGamma, 0);
    int N = PyArray_DIM(pGamma, 1);
    const double *gamma = (double*)PyArray_DATA(pGamma);
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp nomB_dims[2] = {N,M};
    PyObject *pNomB = PyArray_ZEROS(2, nomB_dims, NPY_DOUBLE, 0);

    double *nomB = (double*) PyArray_DATA(pNomB);
    compute_nomB(nomB, gamma, O, N, M, T);

    return pNomB;
}

static PyObject *
py_xi(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO, *pAlpha, *pBeta;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
            &PyArray_Type, &pAlpha,
            &PyArray_Type, &pBeta,
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO))
    {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const double *A  = (double*)PyArray_DATA(pA);
    const double *B  = (double*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    const double *beta = (double*)PyArray_DATA(pBeta);
    const double *alpha = (double*)PyArray_DATA(pAlpha);

    npy_intp xi_dims[2] = {N,N};
    PyObject *pXi = PyArray_ZEROS(2, xi_dims, NPY_DOUBLE, 0);

    double *xi = (double*) PyArray_DATA(pXi);
    computeXi(xi, A, B, O, alpha, beta, N, M, T);
    
    return pXi;
}


static PyObject *
py_update_mult(PyObject *self, PyObject *args)
{
    PyArrayObject *pWeights, *pNomsA, *pDenomsA, *pNomsB, *pDenomsB;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
            &PyArray_Type, &pWeights,
            &PyArray_Type, &pNomsA,
            &PyArray_Type, &pDenomsA,
            &PyArray_Type, &pNomsB,
            &PyArray_Type, &pDenomsB))
    {
        return NULL;
    }
    int K = PyArray_DIM(pWeights, 0);
    int N = PyArray_DIM(pNomsA, 0);
    int M = PyArray_DIM(pNomsB, 1);
    const double *weights = (double*)PyArray_DATA(pWeights);
    const double *nomsA = (double*)PyArray_DATA(pNomsA);
    const double *denomsA = (double*)PyArray_DATA(pDenomsA);
    const double *nomsB = (double*)PyArray_DATA(pNomsB);
    const double *denomsB = (double*)PyArray_DATA(pDenomsB);
    
    npy_intp A_dims[2] = {N,N};
    npy_intp B_dims[2] = {N,M};
    PyObject *pA = PyArray_ZEROS(2, A_dims, NPY_DOUBLE, 0);
    PyObject *pB = PyArray_ZEROS(2, B_dims, NPY_DOUBLE, 0);

    double *A  = (double*) PyArray_DATA(pA);
    double *B  = (double*) PyArray_DATA(pB);
    update_multiple(A, B, weights, nomsA, denomsA, nomsB, denomsB, N, M, K);

    PyObject *tuple = Py_BuildValue("(O,O)", pA, pB);
    Py_DECREF(pA);
    Py_DECREF(pB);
    return tuple;
}


static PyObject *
py_update(PyObject *self, PyObject *args)
{
    PyArrayObject *pO, *pXi, *pGamma;
    int M;
    if (!PyArg_ParseTuple(args, "O!O!O!i",
            &PyArray_Type, &pO,
            &PyArray_Type, &pGamma,
            &PyArray_Type, &pXi,
            &M))
    {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pGamma, 1);
    const short *O = (short*)PyArray_DATA(pO);
    const double *gamma = (double*)PyArray_DATA(pGamma);
    const double *xi    = (double*)PyArray_DATA(pXi);
    
    npy_intp A_dims[2] = {N,N};
    npy_intp B_dims[2] = {N,M};
    npy_intp pi_dims[1] = {N};
    PyObject *pA = PyArray_ZEROS(2, A_dims, NPY_DOUBLE, 0);
    PyObject *pB = PyArray_ZEROS(2, B_dims, NPY_DOUBLE, 0);
    PyObject *pPi = PyArray_ZEROS(1, pi_dims, NPY_DOUBLE, 0);

    double *A  = (double*) PyArray_DATA(pA);
    double *B  = (double*) PyArray_DATA(pB);
    double *pi = (double*) PyArray_DATA(pPi);
    update(A, B, pi, O, gamma, xi, N, M, T);

    PyObject *tuple = Py_BuildValue("(O,O,O)", pA, pB, pPi);
    Py_DECREF(pA);
    Py_DECREF(pB);
    Py_DECREF(pPi);
    return tuple;
}

static PyObject *
py_forward_no_scaling32(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pPi, *pO;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pPi,
            &PyArray_Type, &pO)) {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const float *A  = (float*)PyArray_DATA(pA);
    const float *B  = (float*)PyArray_DATA(pB);
    const float *pi = (float*)PyArray_DATA(pPi);
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp alpha_dims[2] = {T, N};
    PyObject *pAlpha = PyArray_ZEROS(2, alpha_dims, NPY_FLOAT32, 0);
    
    float *alpha = (float*) PyArray_DATA(pAlpha);
    float logprob = forward_no_scaling32(alpha, A, B, pi, O, N, M, T);
    
    PyObject *tuple = Py_BuildValue("(d,O)", logprob, pAlpha);
    Py_DECREF(pAlpha);
    return tuple;
}

static PyObject *
py_forward32(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pPi, *pO;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pPi,
            &PyArray_Type, &pO)) {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const float *A  = (float*)PyArray_DATA(pA);
    const float *B  = (float*)PyArray_DATA(pB);
    const float *pi = (float*)PyArray_DATA(pPi);
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp alpha_dims[2] = {T, N};
    npy_intp scale_dims[1] = {T};
    PyObject *pAlpha = PyArray_ZEROS(2, alpha_dims, NPY_FLOAT32, 0);
    PyObject *pScale = PyArray_ZEROS(1, scale_dims, NPY_FLOAT32, 0);
    
    float *alpha = (float*) PyArray_DATA(pAlpha);
    float *scale = (float*) PyArray_DATA(pScale);
    float logprob = forward32(alpha, scale, A, B, pi, O, N, M, T);
    PyObject *tuple = Py_BuildValue("(f,O,O)", logprob, pAlpha, pScale);
    Py_DECREF(pAlpha);
    Py_DECREF(pScale);
    return tuple;
}

static PyObject *
py_backward32(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO, *pScale;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO,
            &PyArray_Type, &pScale)) {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const float *A  = (float*)PyArray_DATA(pA);
    const float *B  = (float*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    const float *scale = (float*)PyArray_DATA(pScale);
    
    npy_intp dims[2] = {T, N};
    PyObject *pBeta = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    
    float *beta = (float*) PyArray_DATA(pBeta);
    backward32(beta, A, B, O, scale, N, M, T);

    return pBeta;
}

static PyObject *
py_backward_no_scaling32(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO;
    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO)) {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const float *A  = (float*)PyArray_DATA(pA);
    const float *B  = (float*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp dims[2] = {T, N};
    PyObject *pBeta = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    
    float *beta = (float*) PyArray_DATA(pBeta);
    backward_no_scaling32(beta, A, B, O, N, M, T);

    return pBeta;
}

static PyObject *
py_gamma32(PyObject *self, PyObject *args)
{
    PyArrayObject *pAlpha, *pBeta;
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &pAlpha,
            &PyArray_Type, &pBeta)) {
        return NULL;
    }
    int T = PyArray_DIM(pAlpha, 0);
    int N = PyArray_DIM(pAlpha, 1);
    const float *alpha = (float*)PyArray_DATA(pAlpha);
    const float *beta  = (float*)PyArray_DATA(pBeta);
    
    npy_intp dims[2] = {T, N};
    PyObject *pGamma = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);

    float *gamma = (float*) PyArray_DATA(pGamma);
    computeGamma32(gamma, alpha, beta, T, N);

    return pGamma;
}

static PyObject *
py_nomA32(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO, *pAlpha, *pBeta;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
            &PyArray_Type, &pAlpha,
            &PyArray_Type, &pBeta,
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO))
    {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const float *A  = (float*)PyArray_DATA(pA);
    const float *B  = (float*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    const float *beta = (float*)PyArray_DATA(pBeta);
    const float *alpha = (float*)PyArray_DATA(pAlpha);
    
    npy_intp nomA_dims[2] = {N,N};
    PyObject *pNomA = PyArray_ZEROS(2, nomA_dims, NPY_FLOAT32, 0);

    float *nomA = (float*) PyArray_DATA(pNomA);
    compute_nomA32(nomA, A, B, O, alpha, beta, N, M, T);
    
    return pNomA;
}

static PyObject *
py_denomA32(PyObject *self, PyObject *args)
{
    PyArrayObject *pGamma;
    int T;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &pGamma, &T))
    {
        return NULL;
    }
    int N = PyArray_DIM(pGamma, 1);
    const float *gamma = (float*)PyArray_DATA(pGamma);
    
    npy_intp denomA_dims[1] = {N};
    PyObject *pDenomA = PyArray_ZEROS(1, denomA_dims, NPY_FLOAT32, 0);

    float *denomA = (float*) PyArray_DATA(pDenomA);
    compute_denomA32(denomA, gamma, T, N);

    return pDenomA;
}

static PyObject *
py_nomB32(PyObject *self, PyObject *args)
{
    PyArrayObject *pO, *pGamma;
    int M;
    if (!PyArg_ParseTuple(args, "O!O!i",
            &PyArray_Type, &pO,
            &PyArray_Type, &pGamma,
            &M))
    {
        return NULL;
    }
    int T = PyArray_DIM(pGamma, 0);
    int N = PyArray_DIM(pGamma, 1);
    const float *gamma = (float*)PyArray_DATA(pGamma);
    const short *O = (short*)PyArray_DATA(pO);
    
    npy_intp nomB_dims[2] = {N,M};
    PyObject *pNomB = PyArray_ZEROS(2, nomB_dims, NPY_FLOAT32, 0);

    float *nomB = (float*) PyArray_DATA(pNomB);
    compute_nomB32(nomB, gamma, O, N, M, T);

    return pNomB;
}

static PyObject *
py_xi32(PyObject *self, PyObject *args)
{
    PyArrayObject *pA, *pB, *pO, *pAlpha, *pBeta;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
            &PyArray_Type, &pAlpha,
            &PyArray_Type, &pBeta,
            &PyArray_Type, &pA,
            &PyArray_Type, &pB,
            &PyArray_Type, &pO))
    {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pA, 0);
    int M = PyArray_DIM(pB, 1);
    const float *A  = (float*)PyArray_DATA(pA);
    const float *B  = (float*)PyArray_DATA(pB);
    const short *O = (short*)PyArray_DATA(pO);
    const float *beta = (float*)PyArray_DATA(pBeta);
    const float *alpha = (float*)PyArray_DATA(pAlpha);

    npy_intp xi_dims[2] = {N,N};
    PyObject *pXi = PyArray_ZEROS(2, xi_dims, NPY_FLOAT32, 0);

    float *xi = (float*) PyArray_DATA(pXi);
    computeXi32(xi, A, B, O, alpha, beta, N, M, T);
    
    return pXi;
}


static PyObject *
py_update_mult32(PyObject *self, PyObject *args)
{
    PyArrayObject *pWeights, *pNomsA, *pDenomsA, *pNomsB, *pDenomsB;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
            &PyArray_Type, &pWeights,
            &PyArray_Type, &pNomsA,
            &PyArray_Type, &pDenomsA,
            &PyArray_Type, &pNomsB,
            &PyArray_Type, &pDenomsB))
    {
        return NULL;
    }
    int K = PyArray_DIM(pWeights, 0);
    int N = PyArray_DIM(pNomsA, 0);
    int M = PyArray_DIM(pNomsB, 1);
    const float *weights = (float*)PyArray_DATA(pWeights);
    const float *nomsA = (float*)PyArray_DATA(pNomsA);
    const float *denomsA = (float*)PyArray_DATA(pDenomsA);
    const float *nomsB = (float*)PyArray_DATA(pNomsB);
    const float *denomsB = (float*)PyArray_DATA(pDenomsB);
    
    npy_intp A_dims[2] = {N,N};
    npy_intp B_dims[2] = {N,M};
    PyObject *pA = PyArray_ZEROS(2, A_dims, NPY_FLOAT32, 0);
    PyObject *pB = PyArray_ZEROS(2, B_dims, NPY_FLOAT32, 0);

    float *A  = (float*) PyArray_DATA(pA);
    float *B  = (float*) PyArray_DATA(pB);
    update_multiple32(A, B, weights, nomsA, denomsA, nomsB, denomsB, N, M, K);

    PyObject *tuple = Py_BuildValue("(O,O)", pA, pB);
    Py_DECREF(pA);
    Py_DECREF(pB);
    return tuple;
}


static PyObject *
py_update32(PyObject *self, PyObject *args)
{
    PyArrayObject *pO, *pXi, *pGamma;
    int M;
    if (!PyArg_ParseTuple(args, "O!O!O!i",
            &PyArray_Type, &pO,
            &PyArray_Type, &pGamma,
            &PyArray_Type, &pXi,
            &M))
    {
        return NULL;
    }
    int T = PyArray_DIM(pO, 0);
    int N = PyArray_DIM(pGamma, 1);
    const short *O = (short*)PyArray_DATA(pO);
    const float *gamma = (float*)PyArray_DATA(pGamma);
    const float *xi    = (float*)PyArray_DATA(pXi);
    
    npy_intp A_dims[2] = {N,N};
    npy_intp B_dims[2] = {N,M};
    npy_intp pi_dims[1] = {N};
    PyObject *pA = PyArray_ZEROS(2, A_dims, NPY_FLOAT32, 0);
    PyObject *pB = PyArray_ZEROS(2, B_dims, NPY_FLOAT32, 0);
    PyObject *pPi = PyArray_ZEROS(1, pi_dims, NPY_FLOAT32, 0);

    float *A  = (float*) PyArray_DATA(pA);
    float *B  = (float*) PyArray_DATA(pB);
    float *pi = (float*) PyArray_DATA(pPi);
    update32(A, B, pi, O, gamma, xi, N, M, T);

    PyObject *tuple = Py_BuildValue("(O,O,O)", pA, pB, pPi);
    Py_DECREF(pA);
    Py_DECREF(pB);
    Py_DECREF(pPi);
    return tuple;
}

/***
 * structure to describe our methods to python.
 */
static PyMethodDef HmmMethods[] = {
        {"forward", py_forward, METH_VARARGS, forward_doc},
        {"forward32", py_forward32, METH_VARARGS, forward_doc},
        {"backward", py_backward, METH_VARARGS, backward_doc},
        {"backward32", py_backward32, METH_VARARGS, backward_doc},
        {"forward_no_scaling", py_forward_no_scaling, METH_VARARGS, forward_doc},
        {"forward_no_scaling32", py_forward_no_scaling32, METH_VARARGS, forward_doc},
        {"backward_no_scaling", py_backward_no_scaling, METH_VARARGS, backward_doc},
        {"backward_no_scaling32", py_backward_no_scaling32, METH_VARARGS, backward_doc},
        {"update", py_update, METH_VARARGS, update_model_doc},
        {"update32", py_update32, METH_VARARGS, update_model_doc},
        {"state_probabilities", py_gamma, METH_VARARGS, compute_gamma_doc},
        {"state_probabilities32", py_gamma32, METH_VARARGS, compute_gamma_doc},
        {"state_counts", py_denomA, METH_VARARGS, compute_xi_doc},    
        {"state_counts32", py_denomA32, METH_VARARGS, compute_xi_doc},    
        {"symbol_counts", py_nomB, METH_VARARGS, compute_xi_doc},
        {"symbol_counts32", py_nomB32, METH_VARARGS, compute_xi_doc},
        {"transition_probabilities", py_xi, METH_VARARGS, compute_xi_doc},
        {"transition_probabilities32", py_xi32, METH_VARARGS, compute_xi_doc},
        {"transition_counts", py_nomA, METH_VARARGS, compute_xi_doc},
        {"transition_counts32", py_nomA32, METH_VARARGS, compute_xi_doc},
        {NULL, NULL, 0, NULL}
};

/***
 * This is the intializer for our functions. Technical stuff
 */
PyMODINIT_FUNC
initc(void) {
    (void) Py_InitModule("c", HmmMethods);
    import_array();
}

