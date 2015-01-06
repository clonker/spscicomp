import hmm.kernel.simple as simple
import unittest
import numpy as np

# Observation Sequence

ob = np.array([1, 0, 1, 1, 0, 1, 1, 1])

# model parameters

A = np.array(
    [[ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ]])

B = np.array(
    [[ 1.0, 0.0 ],
     [ 0.5, 0.5 ],
     [ 0.0, 1.0 ]])

pi = np.array([ 0.333, 0.333, 0.333])

class TestForward(unittest.TestCase):

	def test_induction_no_scaling(self):
		T, N = len(ob), len(A)
		_, alpha = simple.forward_no_scaling(A, B, pi, ob)

		# initial condition
		for i in range(1,N):
			self.assertAlmostEqual(alpha[0,i], pi[i]*B[i, ob[0]])

		# induction
		for t in range(1,T):
			for j in range(1,N):
				self.assertAlmostEqual(
					alpha[t,j],
					np.sum(alpha[t-1,:]*A[:,j]) * B[j,ob[t]])

if __name__ == '__main__':
    unittest.main()
