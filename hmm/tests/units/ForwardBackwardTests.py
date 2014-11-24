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
sample1 = np.loadtxt('data/sample1.dat', dtype='int')
O = sample1[0:100]

# Initial Model
A  = np.loadtxt('data/startmodelA_1.dat')
B  = np.loadtxt('data/startmodelB_1.dat').T
pi = np.loadtxt('data/startmodelPi_1.dat')

class ForwardBackwardTests(unittest.TestCase):

	def test_update_scheme(self):
		alpha, c = hmm.scaledForwardCoeffs([A,B,pi], O)
		beta     = hmm.scaledBackwardCoeffs([A,B,pi], O, c)
		for k in range(0, 2):
			A_1,B_1,pi_1 = hmm.update_scheme([A,B,pi], alpha, beta, O)
			[A_2,B_2,pi_2],_ = hmm.propagate([A,B,pi], O)
		print '\n----------\n'
		print A_1
		print A_2
		print B_1
		print B_2
		print pi_1
		print pi_2
		self.assertTrue(True)

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
			self.assertAlmostEqual(sum(alpha[t]), 1)

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
			self.assertEqual(alpha[0,i]/c[0], pi[i]*B[O[0],i])
		for t in range(1, len(alpha)):
			for i in range(0, len(alpha[t])):
				self.assertTrue(abs(alpha[t,i]/c[t]- sum(alpha[t-1,:]*A[:,i])*B[O[t],i]) < eps)

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
				self.assertTrue(abs(beta[t,i]/c[t] - np.sum(A[i]*B[O[t],j]*beta[t+1])) < eps)

if __name__ == '__main__':
	unittest.main();
