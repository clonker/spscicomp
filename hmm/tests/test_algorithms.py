import hmm.algorithms
import unittest
import numpy

ob = numpy.array([1, 0, 1, 1, 0, 1, 1, 0])

T = len(ob)

# initial conditions
A = numpy.array(
    [[ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ]])

B = numpy.array(
    [[ 1.0, 0.0 ],
     [ 0.5, 0.5 ],
     [ 0.0, 1.0 ]])

pi = numpy.array([ 0.333, 0.333, 0.333])

N = len(pi)

M = len(B[0])

class TestBaumWelch:

	def test_if_it_even_runs(self):
		A, B, pi, eps, it = hmm.algorithms.baum_welch(ob, A, B, pi)
		print A, B, pi, eps 
		self.assertTrue(it > 0)

if __name__ == '__main__':
    unittest.main()