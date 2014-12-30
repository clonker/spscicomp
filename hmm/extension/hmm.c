#include "hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double forward(
		const double *A,
		const double *B,
		const double *pi,
		const long *O,
		int N, int M, int T,
		double *alpha,
		double *scale)
{
	int i, j, t;
	double sum, logprob;

	scale[0] = 0.0;
	for (i = 0; i < N; i++) {
		alpha[i]  = pi[i] * B[i*M+O[0]];
		scale[0] += alpha[i];
	}
	if (scale[0] != 0)
	for (i = 0; i < N; i++)
		alpha[i] /= scale[0];
	for (t = 0; t < T-1; t++) {
		scale[t+1] = 0.0;
		for (j = 0; j < N; j++) {
			sum = 0.0;
			for (i = 0; i < N; i++) {
				sum += alpha[t*N+i]*A[i*N+j];
			}
			alpha[(t+1)*N+j] = sum * B[j*M+O[t+1]];
			scale[t+1] += alpha[(t+1)*N+j];
		}
		if (scale[t+1] != 0)
		for (j = 0; j < N; j++)
			alpha[(t+1)*N+j] /= scale[t+1];
	}
	// calculate likelihood
	logprob = 0.0;
	for (t = 0; t < T; t++)
		logprob += log(scale[t]);
	return logprob;
}

int backward(
		const double *A,
		const double *B,
		const double *pi,
		const long *O,
		int N, int M, int T,
		double *beta,
		const double *scale)
{
	int i, j, t;
	double sum;

	for (i = 0; i < N; i++)
		if (scale[T-1] != 0)
			beta[(T-1)*N+i] = 1.0 / scale[T-1];
		else
			beta[(T-1)*N+1] = 1.0;
	for (t = T-2; t >= 0; t--)
		for (i = 0; i < N; i++) {
			sum = 0.0;
			for (j = 0; j < N; j++)
				sum += A[i*N+j]*B[j*M+O[t+1]]*beta[(t+1)*N+j];
			if (scale[t] != 0)
				beta[t*N+i] = sum / scale[t];
			else
				beta[t*N+i] = sum;
		}
	return EXIT_SUCCESS;
}

int computeGamma(
		const double *alpha,
		const double *beta,
		int T, int N,
		double *gamma)
{
	int i, t;
	double sum;

	for (t = 0; t < T; t++) {
		sum = 0.0;
		for (i = 0; i < N; i++) {
			gamma[t*N+i] = alpha[t*N+i]*beta[t*N+i];
			sum += gamma[t*N+i];
		}
		for (i = 0; i < N; i++)
			gamma[t*N+i] /= sum;
	}
	return EXIT_SUCCESS;
}

int computeXi(
		const double *A,
		const double *B,
		const double *pi,
		const long *O,
		int N, int M, int T,
		const double *alpha,
		const double *beta,
		double *xi)
{
	int i, j, t;
	double sum;

	for (t = 0; t < T-1; t++) {
		sum = 0.0;
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++) {
				xi[t*N*N + i*N + j] = alpha[t*N+i]*beta[(t+1)*N+j]*A[i*N+j]*B[j*M+O[t+1]];
				sum += xi[t*N*N + i*N + j];
			}
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++)
				xi[t*N*N + i*N + j] /= sum;
	}
	return EXIT_SUCCESS;
}

int update(
		double *A,
		double *B,
		double *pi,
		const long *O,
		int N, int M, int T,
		const double *gamma,
		const double *xi)
{
	int i, j, k, t;
	double gamma_sum, sum;

	/* UPDATE INITIAL CONDITION */
	for (i = 0; i < N; i++)
		pi[i] = gamma[i];

	for (i = 0; i < N; i++) {
		gamma_sum = 0.0;
		for (t = 0; t < T-1; t++)
			gamma_sum += gamma[t*N+i];

		/* UPDATE TRANSITION MATRIX A */
		for (j = 0; j < N; j++) {
			sum = 0.0;
			for (t = 0; t < T-1; t++)
				sum += xi[t*N*N + i*N + j];
			A[i*N + j] = sum / gamma_sum;
		}

		/* UPDATE SYMBOL PROBABILITY B */
		gamma_sum += gamma[(T-1)*N + i];
		for (k = 0; k < M; k++) {
			sum = 0.0;
			for (t = 0; t < T; t++)
				if (O[t] == k)
					sum += gamma[t*N + i];
			B[i*M + k] = sum / gamma_sum;
		}
	}
	return EXIT_SUCCESS;
}
