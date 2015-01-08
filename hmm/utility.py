<<<<<<< HEAD
import numpy 
import hmm.kernel.python

def random_sequence(A, B, pi, T, kernel=hmm.kernel.python):
	obs = kernel.random_sequence(A, B, pi, T)
	return obs

def compare_models(A1, B1, pi1, A2, B2, pi2, T, kernel=hmm.kernel.python):
	""" Give a measure for the similarity of two models."""
	obs = kernel.random_sequence(A2, B2, pi2, T)
	logprob1, _, _ = kernel.forward(A1, B1, pi1, obs)
	logprob2, _, _ = kernel.forward(A2, B2, pi2, obs)
	similarity1 = (logprob2 - logprob1) / float(T)
	obs = kernel.random_sequence(A1, B1, pi1, T)
	logprob1, _, _ = kernel.forward(A1, B1, pi1, obs)
	logprob2, _, _ = kernel.forward(A2, B2, pi2, obs)
	similarity2 = (logprob2 - logprob1) / float(T)
	return 0.5 * (similarity1 + similarity2)
=======
import numpy

def generate_sequence(A, B, pi, T):
    """Creates a random Sequence of length n on base of this model."""
    ob = numpy.empty(T, dtype=numpy.int16)
    current = random_by_dist(pi)
    for t in xrange(T):
        ob[t]   = random_by_dist(B[current,:])
        current = random_by_dist(A[current])
    return ob

def random_by_dist(distribution):
    x = numpy.random.random();
    for n in xrange(len(distribution)):
        if x < (distribution[n]):
            return n;
        else:
            x -= distribution[n];
    return n
>>>>>>> 05f862519fba04175e35aff7b0e9b0b690dfd395
