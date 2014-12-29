#ifndef HMM_H_
#define HMM_H_

typedef struct {
	double *A;
	double *B;
	double *pi;
	int N;
	int M;
} HMM;

double forward(
		const double *A,
		const double *B,
		const double *pi,
		const long *O,
		int N, int M, int T,
		double *alpha,
		double *scale);

int backward(
		const double *A,
		const double *B,
		const double *pi,
		const long *O,
		int N, int M, int T,
		double *beta,
		const double *scale);

int computeGamma(
		const double *alpha,
		const double *beta,
		int T, int N,
		double *gamma);

int computeXi(
		const double *A,
		const double *B,
		const double *pi,
		const long *O,
		int N, int M, int T,
		const double *alpha,
		const double *beta,
		double *xi);

int update(
		double *A,
		double *B,
		double *pi,
		const long *O,
		int N, int M, int T,
		const double *gamma,
		const double *xi);


#endif /* HMM_H_ */
