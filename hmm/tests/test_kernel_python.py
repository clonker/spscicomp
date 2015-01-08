import hmm.kernel.python
import hmm.models
import unittest
import numpy

# Observation Sequence

class TestCounts(unittest.TestCase):

    kernel = hmm.kernel.python  
    ob = [1, 0, 1, 1]
    A, B, pi = hmm.models.get_models()['equi32']

    def test_symbol_counts_is_same_as_per_hand(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)
        symbol_counts = self.kernel.symbol_counts(alpha, beta, self.ob, len(self.B[0]))
        gamma = self.kernel.state_probabilities(alpha, beta)
        B  = numpy.zeros((len(self.B),len(self.B[0])))
        for i in range(len(B)):
            for k in range(len(B[0])):
                B[i,k] = 0.0
                for t in range(len(self.ob)):
                    if self.ob[t] == k:
                        B[i,k] += gamma[t,i]
        numpy.testing.assert_almost_equal(B, symbol_counts)

    def test_state_counts_is_same_as_sum_of_state_probs(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)
        gamma = self.kernel.state_probabilities(alpha, beta)
        gamma_counts = self.kernel.state_counts(alpha, beta, len(alpha))

        numpy.testing.assert_almost_equal(numpy.sum(gamma, axis=0), gamma_counts)

    def test_transition_counts_is_same_as_sum_of_transition_probs(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)
        xi = self.kernel.transition_probabilities(alpha, beta, self.A, self.B, self.ob)
        xi_counts = self.kernel.transition_counts(alpha, beta, self.A, self.B, self.ob)

        numpy.testing.assert_almost_equal(numpy.sum(xi, axis=0), xi_counts)


class TestScaling(unittest.TestCase):

    kernel = hmm.kernel.python  
    ob = [1, 0, 1, 1]
    A, B, pi = hmm.models.get_models()['equi32']

    def test_transition_counts_independ(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)
        xi = self.kernel.transition_counts(alpha, beta, self.A, self.B, self.ob)

        _, alpha_s, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta_s = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        xi_s = self.kernel.transition_counts(alpha_s, beta_s, self.A, self.B, self.ob)

        numpy.testing.assert_almost_equal(xi, xi_s)

    def test_symbol_counts_independ(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)
        gamma = self.kernel.symbol_counts(alpha, beta, self.ob, len(self.B[0]))

        _, alpha_s, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta_s = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        gamma_s = self.kernel.symbol_counts(alpha, beta, self.ob, len(self.B[0]))

        numpy.testing.assert_almost_equal(gamma, gamma_s)

    def test_transition_probs_independ(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)
        xi = self.kernel.transition_probabilities(alpha, beta, self.A, self.B, self.ob)

        _, alpha_s, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta_s = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        xi_s = self.kernel.transition_probabilities(alpha_s, beta_s, self.A, self.B, self.ob)

        numpy.testing.assert_almost_equal(xi, xi_s)

    def test_symbol_probs_independ(self):
        _, alpha = self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)
        gamma = self.kernel.state_probabilities(alpha, beta)

        _, alpha_s, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta_s = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        gamma_s = self.kernel.state_probabilities(alpha_s, beta_s)

        numpy.testing.assert_almost_equal(gamma, gamma_s)


class TestForward(unittest.TestCase):
    ob = numpy.array([1, 0, 1, 1, 0, 1, 1, 1])

    def test_induction_no_scaling(self):
        A, B, pi = hmm.models.get_models()['equi32']
        ob = self.ob
        T, N = len(ob), len(A)
        _, alpha = hmm.kernel.python.forward_no_scaling(A, B, pi, ob)
        # initial condition
        for i in range(1,N):
            self.assertAlmostEqual(alpha[0,i], pi[i]*B[i, ob[0]])
        # induction
        for t in range(1,T):
            for j in range(1,N):
                self.assertAlmostEqual(
                    alpha[t,j],
                    numpy.sum(alpha[t-1,:]*A[:,j]) * B[j,ob[t]])

class TestCallErrors(unittest.TestCase):

    kernel = hmm.kernel.python  
    ob = [1, 0, 1, 1]
    A, B, pi = hmm.models.get_models()['equi32']

    def test_forward(self):
        self.kernel.forward(self.A, self.B, self.pi, self.ob)

    def test_forward_no_scaling(self):
        self.kernel.forward_no_scaling(self.A, self.B, self.pi, self.ob)

    def test_backward_no_scaling(self):
        self.kernel.backward_no_scaling(self.A, self.B, self.pi, self.ob)

    def test_backward(self):
        _, _, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)

    def test_state_probabilities(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        self.kernel.state_probabilities(alpha, beta)

    def test_state_counts(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        self.kernel.state_counts(alpha, beta, 3)

    def test_symbol_counts(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        self.kernel.symbol_counts(alpha, beta, self.ob, len(self.B[0]))

    def test_transition_probabilities(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        self.kernel.transition_probabilities(alpha, beta, self.A, self.B, self.ob)

    def test_transition_counts(self):
        _, alpha, scaling = self.kernel.forward(self.A, self.B, self.pi, self.ob)
        beta = self.kernel.backward(self.A, self.B, self.pi, self.ob, scaling)
        self.kernel.transition_counts(alpha, beta, self.A, self.B, self.ob)




if __name__ == '__main__':
    unittest.main()
