import hmm.kernel.python
import hmm.utility
import unittest
import numpy

# Observation Sequence

class TestCounts(unittest.TestCase):

    kernel = hmm.kernel.python
    ob = numpy.array([1, 0, 1, 1])
    A, B, pi = hmm.utility.get_models()['equi32']
    dtype = numpy.float32

    def test_symbol_counts_is_same_as_per_hand(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.ob, self.dtype)
        gamma = self.kernel.state_probabilities(alpha, beta, self.dtype)
        symbol_counts = self.kernel.symbol_counts(gamma, self.ob, len(self.B[0]), self.dtype)
        B  = numpy.zeros((len(self.B),len(self.B[0])), self.dtype)
        for i in range(len(B)):
            for k in range(len(B[0])):
                B[i,k] = 0.0
                for t in range(len(self.ob)):
                    if self.ob[t] == k:
                        B[i,k] += gamma[t,i]
        numpy.testing.assert_almost_equal(B, symbol_counts)

    def test_state_counts_is_same_as_sum_of_state_probs(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.ob, self.dtype)
        gamma = self.kernel.state_probabilities(alpha, beta, self.dtype)
        gamma_counts = self.kernel.state_counts(gamma, len(alpha), self.dtype)

        numpy.testing.assert_almost_equal(numpy.sum(gamma, axis=0), gamma_counts)

    def test_transition_counts_is_same_as_sum_of_transition_probs(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.ob, self.dtype)
        xi = self.kernel.transition_probabilities(alpha, beta, self.A, self.B, self.ob, self.dtype)
        xi_counts = self.kernel.transition_counts(alpha, beta, self.A, self.B, self.ob, self.dtype)

        numpy.testing.assert_almost_equal(numpy.sum(xi, axis=0), xi_counts)


class TestScaling(unittest.TestCase):

    kernel = hmm.kernel.python
    ob = numpy.array([1, 0, 1, 1])
    A, B, pi = hmm.utility.get_models()['equi32']
    dtype = numpy.float32

    def test_transition_counts_independ(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.ob, self.dtype)
        xi = self.kernel.transition_counts(alpha, beta, self.A, self.B, self.ob, self.dtype)

        _, alpha_s, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta_s = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        xi_s = self.kernel.transition_counts(alpha_s, beta_s, self.A, self.B, self.ob, self.dtype)

        numpy.testing.assert_almost_equal(xi, xi_s)

    def test_transition_probs_independ(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.ob, self.dtype)
        xi = self.kernel.transition_probabilities(alpha, beta, self.A, self.B, self.ob, self.dtype)

        _, alpha_s, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta_s = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        xi_s = self.kernel.transition_probabilities(alpha_s, beta_s, self.A, self.B, self.ob, self.dtype)

        numpy.testing.assert_almost_equal(xi, xi_s)

    def test_state_probs_independ(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.ob, self.dtype)
        gamma = self.kernel.state_probabilities(alpha, beta, self.dtype)

        _, alpha_s, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta_s = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        gamma_s = self.kernel.state_probabilities(alpha_s, beta_s, self.dtype)

        numpy.testing.assert_almost_equal(gamma, gamma_s)


class TestForward(unittest.TestCase):
    ob = numpy.array([1, 0, 1, 1, 0, 1, 1, 1])
    A, B, pi = hmm.utility.get_models()['equi32']
    dtype = numpy.float32
    kernel = hmm.kernel.python

    def test_induction_no_scaling(self):
        ob = self.ob
        T, N = len(self.ob), len(self.A)
        p, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)
        # initial condition
        for i in range(1,N):
            self.assertAlmostEqual(alpha[0,i], self.pi[i]*self.B[i, ob[0]])
        # induction
        for t in range(1,T):
            for j in range(1,N):
                self.assertAlmostEqual(
                    alpha[t,j],
                    numpy.sum(alpha[t-1,:]*self.A[:,j]) * self.B[j,ob[t]])

class TestCallErrors(unittest.TestCase):
    dtype = numpy.float64
    kernel = hmm.kernel.python
    ob = numpy.array([1, 0, 1, 1])
    A, B, pi = hmm.utility.get_models()['equi32']

    def test_forward(self):
        self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)

    def test_forward_no_scaling(self):
        self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob, self.dtype)

    def test_backward_no_scaling(self):
        self.kernel.backward_no_scaling(self.A, self.B, self.ob, self.dtype)

    def test_backward(self):
        _, _, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)

    def test_state_probabilities(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        self.kernel.state_probabilities(alpha, beta, self.dtype)

    def test_state_counts(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        gamma = self.kernel.state_probabilities(alpha, beta, self.dtype)
        self.kernel.state_counts(gamma, 3, self.dtype)

    def test_symbol_counts(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        gamma = self.kernel.state_probabilities(alpha, beta, self.dtype)
        self.kernel.symbol_counts(gamma, self.ob, len(self.B[0]), self.dtype)

    def test_transition_probabilities(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        self.kernel.transition_probabilities(alpha, beta, self.A, self.B, self.ob, self.dtype)

    def test_transition_counts(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob, self.dtype)
        beta = self.kernel.backward(self.A, self.B, self.ob, scaling, self.dtype)
        self.kernel.transition_counts(alpha, beta, self.A, self.B, self.ob, self.dtype)


if __name__ == '__main__':
    unittest.main()
