# perform simple unit test for the hmm kernel
# test these basic functions for correctness on a simple model
#
# - hmm.forward
# - hmm.backward

import unittest
import numpy as np
from hmm import *


# Observation Sequence
sample1 = np.loadtxt('../data/sample1.dat', dtype='int')
observation = sample1[0:100]

# Initial Model
A  = np.loadtxt('../data/startmodelA_1.dat')
B  = np.loadtxt('../data/startmodelB_1.dat').T
pi = np.loadtxt('../data/startmodelPi_1.dat')

model = [A,B,pi]

class ForwardBackwardTests(unittest.TestCase):

	def test_scaledForwardCoeffs_is_normed(self):
		""" check if alpha is normed """
		alpha, _ = forward(model, observation)
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
		alpha, c = forward(model, observation)
		O = observation
		for i in range(0, len(alpha[0])):
			self.assertEqual(alpha[0,i]/c[0], pi[i]*B[O[0],i])
		for t in range(1, len(alpha)):
			for i in range(0, len(alpha[t])):
				self.assertAlmostEqual(alpha[t,i], c[t]*sum(alpha[t-1,:]*A[:,i])*B[O[t],i])

	def test_scaledBackwardCoeffs_is_conform(self):
		""" check for the induction formula for beta

		Initialization:
		    	b_T(i) = 1

		Induction:
		    	b_t(i) = sum_j A_ij B_j(O_t+1) b_t+1(j)

		beta is scaled by its sum. This is taken into consideration.
		"""
		_, c = forward(model, observation)
		beta = backward(model, observation, c)
		O, T = observation, len(observation) - 1
		for i in range(0, len(beta[T])):
			self.assertEqual(beta[T,i], c[T])

		for t in range(T-1, -1, -1):
			for i in range(0, len(beta[t])):
				self.assertAlmostEqual(beta[t,i], c[t]*sum(A[i,:]*B[O[t+1],:]*beta[t+1,:]))

if __name__ == '__main__':
	unittest.main();
