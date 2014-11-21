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

	def test_empty_input(self):
		""" check how the functions react for empty input.

		Should return empty lists
		"""
		alpha, c = hmm.scaledForwardCoeffs([A,B,pi], [])
		beta = hmm.scaledBackwardCoeffs([A,B,pi], [], c)
		self.assertTrue(len(alpha) == 0)
		self.assertTrue(len(beta) == 0)
		self.assertTrue(len(c) == 0)

	def test_scaledForwardCoeffs_is_normed(self):
		""" check if alpha is normed """
		alpha, _ = hmm.scaledForwardCoeffs([A,B,pi], O)
		for t in range(0, len(alpha)):
			self.assertTrue(abs(sum(alpha[t])- 1) < eps)

	def test_scaledForwardCoeffs_is_conform(self):
		""" check if the induction formula holds for alpha

		Initialization:
			a_1(i) = pi_1 * B_i(O_1)

		Induction:
			a_t+1(i) = [ sum_i a_t(i) A_ij ] * B_j(O_t+1)

		alpha is scaled by its sum. So it is taken into consideration.
		"""
		alpha, c = hmm.scaledForwardCoeffs([A,B,pi], O)
		for i in range(0, len(alpha[0])):
			self.assertEqual(alpha[0,i]/c[0], pi[i]*B[i, O[0]])
		for t in range(1, len(alpha)):
			for i in range(0, len(alpha[t])):
				self.assertEqual(alpha[t,i]/c[t], sum(alpha[t-1,:]*A[:,i])*B[i, O[t]])

	def test_scaledBackwardCoeffs_is_conform(self):
		""" check for the induction formula for beta

		Initialization:
		    	b_T(i) = 1

		Induction:
		    	b_t(i) = sum_j A_ij B_j(O_t+1) b_t+1(j)

		beta is scaled by its sum. This is taken into consideration.
		"""
		_, c = hmm.scaledForwardCoeffs([A,B,pi],O)
		beta = hmm.scaledBackwardCoeffs([A,B,pi],O,c)
		T = len(O) - 1
		for i in range(0, len(beta[T])):
			self.assertEqual(beta[T,i], c[T])

		for t in range(T-1, 0):
			for i in range(0, len(beta[t])):
				self.assertEqual(beta[t,i]/c[t], np.sum(A[i]*B[j,O[t]]*beta[t+1]))

if __name__ == '__main__':
	unittest.main();
