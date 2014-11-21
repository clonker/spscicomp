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


O = np.array([1,0,1,0,0,1]);           # Observations
A = np.array([[0.5, 0.5],[0.5, 0.5]]); # transfer matrix
B = np.array([[0.3, 0.7],[1., 0.]]);   # symbol probability
pi = np.array([0.5, 0.5]);             # start probability

class SimpleKernelTests(unittest.TestCase):

	def test_scaledForwardCoeffs_alpha_sum_normed(self):
		""" check if alpha is normed """
		alpha, c = hmm.scaledForwardCoeffs([A,B,pi], O)
		for t in range(0, len(alpha)):
			self.assertTrue(abs(sum(alpha[t])- 1) < eps)

	def test_scaledForwardCoeffs_alpha_is_conform(self):
		""" check if the induction formula holds for alpha

		Initialization:
			a_1(i) = pi_1 * b_i(O_1)

		Induction:
			a_t+1(i) = [ sum_i a_t(i) a_ij ] * b_j(O_t+1)

		alpha is scaled by its sum. So it is taken into consideration.
		"""
		alpha, c = hmm.scaledForwardCoeffs([A,B,pi], O);
		for i in range(0, len(alpha[0])):
			self.assertEqual(alpha[0,i]/c[0], pi[i]*B[i, O[0]])
		for t in range(1, len(alpha)):
			for i in range(0, len(alpha[t])):
				self.assertEqual(alpha[t,i]/c[t], sum(alpha[t-1,:]*A[:,i])*B[i, O[t]])

	def test_scaledBackwardCoeffs_sum_normed(self):
		""" check if beta is normed """
		beta, c = hmm.scaledBackwardCoeffs([A,B,pi], 0);
		for t in range(0, len(beta)):
			self.assertTrue(abs(sum(beta[t])-1) < eps)


if __name__ == '__main__':
	unittest.main();
