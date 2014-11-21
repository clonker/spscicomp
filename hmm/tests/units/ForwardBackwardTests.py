# perform simple unit test for the hmm kernel
# test these basic functions for correctness on a simple model
#
# - scaledForwardCoeffs
# - scaledBackwardCoeffs

import unittest
import numpy as np
import hmm

# geforderte Genauigkeit
eps = 1e-8;


# Observation Sequence
O = np.array([1,0,1,0,0,1]);

# Initial Model
# transfer matrix
A = np.array([
	[0.5, 0.5],
	[0.5, 0.5]
]);
# symbol probability
B = np.array([
	[0.3, 0.7],
	[1., 0.]
]);
# initial probability
pi = np.array([0.5, 0.5]);


class ForwardBackwardTests(unittest.TestCase):
	# forward / backward variables and scaling factors c
	a = np.zeros((len(O),2))
	b = np.zeros((len(O),2))
	c = np.zeros((len(O),1))

	def setUp(self):
		""" calculate the forward/backward variables only once """
		self.a, self.c = hmm.scaledForwardCoeffs([A,B,pi], O)
		self.b   = hmm.scaledBackwardCoeffs([A,B,pi], O, self.c)

	def test_scaledForwardCoeffs_alpha_sum_normed(self):
		""" check if alpha is normed """
		for t in range(0, len(self.a)):
			self.assertTrue(abs(sum(self.a[t])- 1) < eps)

	def test_scaledForwardCoeffs_alpha_is_conform(self):
		""" check if the induction formula holds for alpha

		Initialization:
			a_1(i) = pi_1 * B_i(O_1)

		Induction:
			a_t+1(i) = [ sum_i a_t(i) A_ij ] * B_j(O_t+1)

		alpha is scaled by its sum. So it is taken into consideration.
		"""
		for i in range(0, len(self.a[0])):
			self.assertEqual(self.a[0,i]/self.c[0], pi[i]*B[i, O[0]])
		for t in range(1, len(self.a)):
			for i in range(0, len(self.a[t])):
				self.assertEqual(self.a[t,i]/self.c[t], sum(self.a[t-1,:]*A[:,i])*B[i, O[t]])

	def test_scaledBackwardCoeffs_is_conform(self):
		""" check for the induction formula for beta

		Initialization:
		    	b_T(i) = 1

		Induction:
		    	b_t(i) = sum_j A_ij B_j(O_t+1) b_t+1(j)

		beta is scaled by its sum. This is taken into consideration.
		"""
		T = len(O) - 1
		for i in range(0, len(self.b[T])):
			self.assertEqual(self.b[T,i], self.c[T])

		for t in range(T-1, 0):
			for i in range(0, len(self.b[t])):
				self.assertEqual(self.b[t,i]/self.c[t], sum(A[i]*B[j,O[t]]*self.b[t+1]))

if __name__ == '__main__':
	unittest.main();
