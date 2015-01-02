#include "hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float forward32(
		float *alpha,
		float *scale,
		const float *A,
		const float *B,
		const float *pi,
		const long *O,
		int N, int M, int T)
{
	int i, j, t;
	float sum, logprob;

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
	logprob = 0.0f;
	for (t = 0; t < T; t++)
		logprob += log(scale[t]);
	return logprob;
}

void compute_nomA32(
		float *nomA,
		const float *A,
		const float *B,
		const long *O,
		const float *alpha,
		const float *beta,
		int N, int M, int T)
{
	int i, j, t;
	float sum, *tmp;
	
	tmp = (float*) malloc(N*N * sizeof(float));
	for (t = 0; t < T-1; t++) {
		sum = 0.0;
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++) {
				tmp[i*N+j] = alpha[t*N+i]*beta[(t+1)*N+j]*A[i*N+j]*B[j*M+O[t+1]];
				sum += tmp[i*N+j];
			}
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++)
				nomA[i*N+j] += tmp[i*N+j] / sum;
	}
}

void compute_denomA32(
		float *denomA,
		const float *gamma,
		int T, int N)
{
	int i, t;
	for (i = 0; i < N; i++) {
		denomA[i] = 0.0;
		for (t = 0; t < T-1; t++)
			denomA[i] += gamma[t*N+i];
	}
}

void compute_nomB32(
		float *nomB,
		const float *gamma,
		const long *O,
		int N, int M, int T)
{
	int i, k, t;
	for (i = 0; i < N; i++)
		for (k = 0; k < M; k++) {
			nomB[i*M+k] = 0.0;
			for (t = 0; t < T; t++)
				if (O[t] == k)
					nomB[i*M+k] += gamma[t*N+i];
		}
}

void backward32(
		float *beta,
		const float *A,
		const float *B,
		const long *O,
		const float *scale,
		int N, int M, int T)
{
	int i, j, t;
	float sum;

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
}

void computeGamma32(
		float *gamma,
		const float *alpha,
		const float *beta,
		int T, int N)
{
	int i, t;
	float sum;

	for (t = 0; t < T; t++) {
		sum = 0.0;
		for (i = 0; i < N; i++) {
			gamma[t*N+i] = alpha[t*N+i]*beta[t*N+i];
			sum += gamma[t*N+i];
		}
		for (i = 0; i < N; i++)
			gamma[t*N+i] /= sum;
	}
}

void computeXi32(
		float *xi,
		const float *A,
		const float *B,
		const long *O,
		const float *alpha,
		const float *beta,
		int N, int M, int T)
{
	int i, j, t;
	float sum;

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
}

void update_multiple32(
		float *A,
		float *B,
		const float *weights,
		const float *nomsA,
		const float *denomsA,
		const float *nomsB,
		const float *denomsB,
		int N, int M, int K)
{
	printf("test\n");
	int i, j, k;
	float *nomA = (float*) calloc(N*N,sizeof(float));
	float *nomB = (float*) calloc(N*M,sizeof(float));
	for (i = 0; i < N; i++) {
		float denomA = 0.0;
		float denomB = 0.0;
		for (k = 0; k < K; k++) {
			for (j = 0; j < N; j++)
				nomA[i*N + j] += weights[k]*nomsA[k*N*N + i*N + j];
			for (j = 0; j < M; j++) {
				nomB[i*M + j] += weights[k]*nomsB[k*N*M + i*M + j];
			}
			denomA += weights[k]*denomsA[k*N + i];
			denomB += weights[k]*denomsB[k*N + i];
		}
		printf("test i \n");
		for (j = 0; j < N; j++)
			A[i*N + j] = nomA[i*N + j] / denomA;
		for (j = 0; j < M; j++)
			B[i*M + j] = nomB[i*M + j] / denomB;
		printf("test i \n");
	}
	printf("test\n");
}

void update32(
		float *A,
		float *B,
		float *pi,
		const long *O,
		const float *gamma,
		const float *xi,
		int N, int M, int T)
{
	int i, j, k, t;
	float gamma_sum, sum;

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
}
