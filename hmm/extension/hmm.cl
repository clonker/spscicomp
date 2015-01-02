float forward(
		__global float *alpha,
		__global float *scale,
		__global float *A,
		__global float *B,
		__global float *pi,
		__global int *O,
		int N, int M, int T)
{
	int i, j, t;
	float sum, logprob;

	scale[0] = 0.0f;
	for (i = 0; i < N; i++) {
		alpha[i]  = pi[i] * B[i*M+O[0]];
		scale[0] += alpha[i];
	}
	if (scale[0] != 0)
		for (i = 0; i < N; i++)
			alpha[i] /= scale[0];
	for (t = 0; t < T-1; t++) {
		scale[t+1] = 0.0f;
		for (j = 0; j < N; j++) {
			sum = 0.0f;
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

void compute_nomA(
		__global float *nomA,
		__global float *A,
		__global float *B,
		__global int *O,
		__global float *alpha,
		__global float *beta,
		__global float *tmp,
		int N, int M, int T)
{
	int i, j, t;
	float sum;
	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			nomA[i*N+j] = 0.0f;
	
	for (t = 0; t < T-1; t++) {
		sum = 0.0f;
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
		__global float *denomA,
		__global float *gamma,
		int T, int N)
{
	int i, t;
	for (i = 0; i < N; i++) {
		denomA[i] = 0.0f;
		for (t = 0; t < T-1; t++)
			denomA[i] += gamma[t*N+i];
	}
}

void compute_nomB(
		__global float *nomB,
		__global float *gamma,
		__global int *O,
		int N, int M, int T)
{
	int i, k, t;
	for (i = 0; i < N; i++)
		for (k = 0; k < M; k++) {
			nomB[i*M+k] = 0.0f;
			for (t = 0; t < T; t++)
				if (O[t] == k)
					nomB[i*M+k] += gamma[t*N+i];
		}
}

void backward(
		__global float *beta,
		__global float *A,
		__global float *B,
		__global int *O,
		__global float *scale,
		int N, int M, int T)
{
	int i, j, t;
	float sum;

	for (i = 0; i < N; i++)
		if (scale[T-1] != 0)
			beta[(T-1)*N+i] = 1.0f / scale[T-1];
		else
			beta[(T-1)*N+1] = 1.0f;
	for (t = T-2; t >= 0; t--)
		for (i = 0; i < N; i++) {
			sum = 0.0f;
			for (j = 0; j < N; j++)
				sum += A[i*N+j]*B[j*M+O[t+1]]*beta[(t+1)*N+j];
			if (scale[t] != 0)
				beta[t*N+i] = sum / scale[t];
			else
				beta[t*N+i] = sum;
		}
}

void computeGamma(
		__global float *gamma,
		__global float *alpha,
		__global float *beta,
		int T, int N)
{
	int i, t;
	float sum;

	for (t = 0; t < T; t++) {
		sum = 0.0f;
		for (i = 0; i < N; i++) {
			gamma[t*N+i] = alpha[t*N+i]*beta[t*N+i];
			sum += gamma[t*N+i];
		}
		for (i = 0; i < N; i++)
			gamma[t*N+i] /= sum;
	}
}

__kernel void process_obs(
		__global float *weights,
		__global float *nomsA,
		__global float *denomsA,
		__global float *nomsB,
		__global float *denomsB,
		__global float *A,
		__global float *B,
		__global float *pi,
		__global int *O,
		int N,
		int M,
		int maxT,
		__global int *T,
		__global float *alpha,
		__global float *scale,
		__global float *beta,
		__global float *gamma,
		__global float *nomA)
{
	int i; //, k;
	int gid = get_global_id(0);
	size_t Tg = T[gid];
	float weight;
	
	__global float *_alpha = &alpha[gid*maxT*N];
	__global float *_scale = &scale[gid*maxT];
	__global float *_beta  = &beta[gid*maxT*N];
	__global float *_gamma = &gamma[gid*maxT*N];
	__global float *_nomsA = &nomsA[gid*N*N];
	__global float *_nomA  = &nomA[gid*N*N];
	__global float *_nomsB  = &nomsB[gid*N*M];
	__global float *_denomsA = &denomsA[gid*N];
	__global float *_denomsB = &denomsB[gid*N];
	__global int *_O = &O[gid*Tg];
	

	weight = forward(_alpha, _scale, A, B, pi, _O, N, M, Tg);
	backward(_beta, A, B, _O, _scale, N, M, Tg);
	computeGamma(_gamma, _alpha, _beta, Tg, N);

	compute_nomA(_nomsA, A, B, _O, _alpha, _beta, _nomA, N, M, Tg);
	compute_denomA(_denomsA, gamma, Tg, N);
	for (i = 0; i < N; i++) {
		_denomsB[i] = _denomsA[i] + _gamma[(Tg-1)*N + i];
	}
	compute_nomB(_nomsB, _gamma, _O, N, M, Tg);
	weights[gid] = weight;
} 
