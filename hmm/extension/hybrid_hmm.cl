#define N ${N}
#define M ${M}

__kernel void compute_gamma(
		__global float *alpha,
		__global float *beta,
		__global float *gamma,
		__local float *gamma_t)
{
	size_t t = get_group_id(0);
	size_t i = get_local_id(0);
	int j;
	
	float alpha_ti = alpha[t*N + i];
	float beta_ti  = beta[t*N + i];
	
	gamma_t[i] = alpha_ti * beta_ti;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (i == 0) {
		float sum = 0.0f;
		for (j = 0; j < N; j++)
			sum += gamma_t[j];
		for (j = 0; j < N; j++)
			gamma[t*N+j] = gamma_t[j] / sum;
	}
}

		
