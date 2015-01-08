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

models = dict();

models['hmm_1'] =  (
    numpy.array(
        [[0.8, 0.2, 0.0],
         [0.0, 0.2, 0.8],
         [1.0, 0.0, 0.0]],
    ),
    numpy.array(
        [[1.0, 0.0],
         [0.0, 1.0],
         [0.0, 1.0]],
    ),
    numpy.array(
        [0.5, 0.5, 0.0],
    )
)
models['equi32'] = (
    numpy.array(
        [[ 0.333, 0.333, 0.333 ],
         [ 0.333, 0.333, 0.333 ],
         [ 0.333, 0.333, 0.333 ]], numpy.float32),

    numpy.array(
        [[ 0.5, 0.5 ],
         [ 0.5, 0.5 ],
         [ 0.5, 0.5 ]], numpy.float32),

    numpy.array([ 0.333, 0.333, 0.333 ], numpy.float32)
)
models['equi64'] = (
    numpy.array(
        [[ 0.333, 0.333, 0.333 ],
         [ 0.333, 0.333, 0.333 ],
         [ 0.333, 0.333, 0.333 ]], numpy.float64),

    numpy.array(
        [[ 0.5, 0.5 ],
         [ 0.5, 0.5 ],
         [ 0.5, 0.5 ]], numpy.float64),

    numpy.array([ 0.333, 0.333, 0.333 ], numpy.float64)
)


def get_models():
    return models
