#include "hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double forward(
        double *alpha,
        double *scale,
        const double *A,
        const double *B,
        const double *pi,
        const short *O,
        int N, int M, int T)
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

void compute_nomA(
        double *nomA,
        const double *A,
        const double *B,
        const short *O,
        const double *alpha,
        const double *beta,
        int N, int M, int T)
{
    int i, j, t;
    double sum, *tmp;
    
    tmp = (double*) malloc(N*N * sizeof(double));
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

void compute_denomA(
        double *denomA,
        const double *gamma,
        int T, int N)
{
    int i, t;
    for (i = 0; i < N; i++) {
        denomA[i] = 0.0;
        for (t = 0; t < T-1; t++)
            denomA[i] += gamma[t*N+i];
    }
}

void compute_nomB(
        double *nomB,
        const double *gamma,
        const short *O,
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

void backward(
        double *beta,
        const double *A,
        const double *B,
        const short *O,
        const double *scale,
        int N, int M, int T)
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
}

void computeGamma(
        double *gamma,
        const double *alpha,
        const double *beta,
        int T, int N)
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
}

void computeXi(
        double *xi,
        const double *A,
        const double *B,
        const short *O,
        const double *alpha,
        const double *beta,
        int N, int M, int T)
{
    int i, j, t;
    double sum;
    double *xi_t = (double*) malloc(N*N*sizeof(double));

    for (t = 0; t < T-1; t++) {
        sum = 0.0;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                xi_t[i*N + j] = alpha[t*N+i]*beta[(t+1)*N+j]*A[i*N+j]*B[j*M+O[t+1]];
                sum += xi_t[i*N + j];
            }
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++) {
                xi_t[i*N + j] /= sum;
                xi[i*N + j] += xi_t[i*N + j];
            }
    }
}

void update_multiple(
        double *A,
        double *B,
        const double *weights,
        const double *nomsA,
        const double *denomsA,
        const double *nomsB,
        const double *denomsB,
        int N, int M, int K)
{
    printf("test\n");
    int i, j, k;
    double *nomA = (double*) calloc(N*N,sizeof(double));
    double *nomB = (double*) calloc(N*M,sizeof(double));
    for (i = 0; i < N; i++) {
        double denomA = 0.0;
        double denomB = 0.0;
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

void update(
        double *A,
        double *B,
        double *pi,
        const short *O,
        const double *gamma,
        const double *xi,
        int N, int M, int T)
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
            A[i*N + j] = xi[i*N + j] / gamma_sum;
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
