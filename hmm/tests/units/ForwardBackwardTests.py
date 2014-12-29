# perform simple unit test for the hmm kernel
# test these basic functions for correctness on a simple model
#
# - hmm.forward
# - hmm.backward

import unittest
import numpy as np
from hmm import *


# Observation Sequence
O = np.loadtxt('../data/t1.33333.dat', dtype='int')

# Initial Model
A  = np.loadtxt('../data/test_A.hmm')
B  = np.loadtxt('../data/test_B.hmm')
pi = np.loadtxt('../data/test_pi.hmm')

T = len(O)
N = len(A)

alpha = np.zeros((T,N), dtype=np.double)
beta  = np.zeros((T,N), dtype=np.double)
scale = np.zeros((T), dtype=np.double)

forward(A, B, pi, O, alpha, scale)
backward(A, B, pi, O, beta, scale)


class ForwardBackwardTests(unittest.TestCase):

	def test_scaledForwardCoeffs_is_normed(self):
		""" check if alpha is normed """
		for t in range(0, len(alpha)):
			self.assertAlmostEqual(sum(alpha[t]), 1)

	def test_scaledForwardCoeffs_is_conform(self):
		""" check if the induction formula holds for alpha

		Initialization:
			a_1(i) = pi_1 * B_i(O_1)

		Induction:
			a_t+1(i) = [ sum_i a_t(i) A_ij ] * B_j(O_{t+1})

		alpha is scaled by its sum. So it is taken into consideration.
		"""
		for i in range(0, len(alpha[0])):
			self.assertEqual(alpha[0,i]*scale[0], pi[i]*B[i,O[0]])
		for t in range(1, len(alpha)):
			for i in range(0, len(alpha[t])):
				self.assertAlmostEqual(alpha[t,i]*scale[t], sum(alpha[t-1,:]*A[:,i])*B[i,O[t]])

	def test_scaledBackwardCoeffs_is_conform(self):
		""" check for the induction formula for beta

		Initialization:
		    	b_T(i) = 1

		Induction:
		    	b_t(i) = sum_j A_ij B_j(O_t+1) b_t+1(j)

		beta is scaled by its sum. This is taken into consideration.
		"""
		for i in range(0, len(beta[T-1])):
			self.assertEqual(beta[T-1,i]*scale[T-1], 1)

		for t in range(T-2, -1, -1):
			for i in range(0, len(beta[t])):
				self.assertAlmostEqual(beta[t,i]*scale[t], sum(A[i,:]*B[:,O[t+1]]*beta[t+1,:]))

if __name__ == '__main__':
	unittest.main();
