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