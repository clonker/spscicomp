#include "hmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef __DIMS__
#define __DIMS__
#define DIMM2(arr, i, j)    arr[(i)*M + j]
#define DIM2(arr, i, j)     arr[(i)*N + j]
#define DIM3(arr, t, i , j) arr[(t)*N*N + (i)*N + j]
#define DIMM3(arr, t, i, j) arr[(t)*N*M + (i)*M + j]
#endif

float forward_no_scaling32(
        float *alpha,
        const float *A,
        const float *B,
        const float *pi,
        const short *ob,
        int N, int M, int T)
{
    int i, j, t;
    float sum, logprob = 0.0;

    for (i = 0; i < N; i++)
        DIM2(alpha, 0, i) = pi[i] * DIMM2(B, i, ob[0]);
    for (t = 0; t < T-1; t++)
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (i = 0; i < N; i++)
                sum += DIM2(alpha, t, i) * DIM2(A, i, j);
            DIM2(alpha, t+1, j) = sum * DIMM2(B, j, ob[t+1]);
        }
    for (i = 0; i < N; i++)
        logprob += DIM2(alpha, T-1, i);
    return log(logprob);
}

float forward32(
        float *alpha,
        float *scaling,
        const float *A,
        const float *B,
        const float *pi,
        const short *O,
        int N, int M, int T)
{
    int i, j, t;
    float sum, logprob;

    scaling[0] = 0.0;
    for (i = 0; i < N; i++) {
        alpha[i]  = pi[i] * B[i*M+O[0]];
        scaling[0] += alpha[i];
    }
    if (scaling[0] != 0)
    for (i = 0; i < N; i++)
        alpha[i] /= scaling[0];
    for (t = 0; t < T-1; t++) {
        scaling[t+1] = 0.0;
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (i = 0; i < N; i++) {
                sum += alpha[t*N+i]*A[i*N+j];
            }
            alpha[(t+1)*N+j] = sum * B[j*M+O[t+1]];
            scaling[t+1] += alpha[(t+1)*N+j];
        }
        if (scaling[t+1] != 0)
        for (j = 0; j < N; j++)
            alpha[(t+1)*N+j] /= scaling[t+1];
    }
    // calculate likelihood
    logprob = 0.0f;
    for (t = 0; t < T; t++)
        logprob += log(scaling[t]);
    return logprob;
}

void compute_nomA32(
        float *nomA,
        const float *A,
        const float *B,
        const short *O,
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
    free(tmp);
}

void compute_denomA32(
        float *denomA,
        const float *gamma,
        int T, int N)
{
    int i, t;
    for (t = 0; t < T; t++)
        for (i = 0; i < N; i++)
            denomA[i] += gamma[t*N+i];
}

void compute_nomB32(
        float *nomB,
        const float *gamma,
        const short *ob,
        int N, int M, int T)
{
    int i, k, t;
    for (t = 0; t < t; i++)
        for (i = 0; i < N; i++)
            DIMM2(nomB, i, ob[t]) += DIM2(gamma, t, i);
}

void backward_no_scaling32(
        float *beta,
        const float *A,
        const float *B,
        const short *ob,
        int N, int M, int T)
{
    int i, j, t;
    float sum;

    for (i = 0; i < N; i++)
        DIM2(beta, T-1, i) = 1.0;

    for (t = T-2; t >= 0; t--)
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < N; j++)
                sum += DIM2(A,i,j)*DIMM2(B,j,ob[t+1])*DIM2(beta,t+1,j);
            DIM2(beta,t,i) = sum;
        }
}

void backward32(
        float *beta,
        const float *A,
        const float *B,
        const short *O,
        const float *scaling,
        int N, int M, int T)
{
    int i, j, t;
    float sum;

    for (i = 0; i < N; i++)
        if (scaling[T-1] != 0)
            beta[(T-1)*N+i] = 1.0 / scaling[T-1];
        else
            beta[(T-1)*N+1] = 1.0;
    for (t = T-2; t >= 0; t--)
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < N; j++)
                sum += A[i*N+j]*B[j*M+O[t+1]]*beta[(t+1)*N+j];
            if (scaling[t] != 0)
                beta[t*N+i] = sum / scaling[t];
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
        const short *O,
        const float *alpha,
        const float *beta,
        int N, int M, int T)
{
    int i, j, t;
    float sum;
    float *xi_t = (float*) malloc(N*N*sizeof(float));

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
    free(xi_t);
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
        const short *O,
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
