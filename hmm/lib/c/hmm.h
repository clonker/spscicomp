#ifndef HMM_H_
#define HMM_H_

double forward(
		double *alpha,
		double *scale,
		const double *A,
		const double *B,
		const double *pi,
		const int *O,
		int N, int M, int T);

void backward(
		double *beta,
		const double *A,
		const double *B,
		const int *O,
		const double *scale,
		int N, int M, int T);

void compute_nomA(
		double *nomA,
		const double *A,
		const double *B,
		const int *O,
		const double *alpha,
		const double *beta,
		int N, int M, int T);

void compute_denomA(
		double *denomA,
		const double *gamma,
		int T, int N);

void compute_nomB(
		double *nomB,
		const double *gamma,
		const int *O,
		int N, int M, int T);

void computeGamma(
		double *gamma,
		const double *alpha,
		const double *beta,
		int T, int N);

void computeXi(
		double *xi,
		const double *A,
		const double *B,
		const int *O,
		const double *alpha,
		const double *beta,
		int N, int M, int T);

void update(
		double *A,
		double *B,
		double *pi,
		const int *O,
		const double *gamma,
		const double *xi,
		int N, int M, int T);

void update_multiple(
		double *A,
		double *B,
		const double *weights,
		const double *nomsA,
		const double *denomsA,
		const double *nomsB,
		const double *denomsB,
		int N, int M, int K);
		
float forward32(
		float *alpha,
		float *scale,
		const float *A,
		const float *B,
		const float *pi,
		const int *O,
		int N, int M, int T);

void backward32(
		float *beta,
		const float *A,
		const float *B,
		const int *O,
		const float *scale,
		int N, int M, int T);

void compute_nomA32(
		float *nomA,
		const float *A,
		const float *B,
		const int *O,
		const float *alpha,
		const float *beta,
		int N, int M, int T);

void compute_denomA32(
		float *denomA,
		const float *gamma,
		int T, int N);

void compute_nomB32(
		float *nomB,
		const float *gamma,
		const int *O,
		int N, int M, int T);

void computeGamma32(
		float *gamma,
		const float *alpha,
		const float *beta,
		int T, int N);

void computeXi32(
		float *xi,
		const float *A,
		const float *B,
		const int *O,
		const float *alpha,
		const float *beta,
		int N, int M, int T);

void update32(
		float *A,
		float *B,
		float *pi,
		const int *O,
		const float *gamma,
		const float *xi,
		int N, int M, int T);

void update_multiple32(
		float *A,
		float *B,
		const float *weights,
		const float *nomsA,
		const float *denomsA,
		const float *nomsB,
		const float *denomsB,
		int N, int M, int K);
		
#endif /* HMM_H_ */
