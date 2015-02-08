import numpy 
import hmm.kernel.python
import os

def random_sequence(A, B, pi, T, kernel=hmm.kernel.python):
    obs = kernel.random_sequence(A, B, pi, T)
    return obs

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
models['t2'] = (
    numpy.array(
        [[0.9, 0.05, 0.05],
         [0.45, 0.1, 0.45],
         [0.45, 0.45, 0.1]], numpy.float32),
    numpy.array(
        [[ 0.5,  0.5],
         [0.75, 0.25],
         [0.25, 0.75]], numpy.float32
    ),
    numpy.array(
        [0.333, 0.333, 0.333], numpy.float32
    ),
)


def get_models():
    return models

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


def get_observation_part(filename, observation_length, observation_count, dtype=numpy.uint16):
    """ reads a part of an array out of a binary numpy-array-file
        
    Parameters
    ----------
    filename : filename which contains a binary numpy-array
    observation_length : number of observationsymbols to read
    observation_count : skip (observation_length * observation_count) symbols
    dtype : dtype of array. Default is numpy.uint16

    Returns
    -------
    the (observation_length * observation_count)-th observationpart in a binary numpy array file
    with the length observation_length

    """
    observation_file = open(filename, 'r')
    observation_file.seek(observation_length * observation_count * numpy.dtype(dtype).itemsize, os.SEEK_SET)
    return numpy.fromfile(observation_file, count=observation_length, dtype=dtype)


def generate_random_matrice(state_count, symbol_count, dtype=numpy.float32):
    """ generate a numpy array with shape (state_count, symbol_count)
    each line is normalized to 1
    useful to generate a random matrice for A or B
    """
    B = numpy.zeros((state_count, symbol_count), dtype=dtype)
    for x in range(state_count):
        rowsum = 0.0
        for y in range(symbol_count):
            r = numpy.random.random()
            rowsum += r
            B[x, y] = r
        B[x] /= rowsum
    return B


def generate_random_array(state_count, dtype=numpy.float32):
    """ generate a numpy array with shape (state_count)
    array is normalized to 1
    useful to generate a random array for pi
    """
    pi = numpy.zeros((state_count), dtype=dtype)
    rowsum = 0.0
    for x in range(state_count):
        r = numpy.random.random()
        rowsum += r
        pi[x] = r
    pi /= rowsum
    return pi


def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program
    taken from: http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
    """

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass


class Tarjan(object):

    def __init__(self, adjacence_matrice):
        self.adjacence_matrice = adjacence_matrice
        self.dfs = numpy.zeros((len(adjacence_matrice)))
        self.lowlink = numpy.zeros((len(adjacence_matrice)))
        self.maxdfs = 0
        self.stack = numpy.array()
        self.unvisited = numpy.array()
        for i in len(adjacence_matrice):
            self.unvisited.append(i)
        while len(self.unvisited)>0:
            tarjan(self.unvisited.pop())

    def tarjan(self, node):
        self.dfs[node] = self.maxdfs
        self.lowlink[node] = self.maxdfs
        self.maxdfs += 1
        
        
        
    

class ChunkedArray(object):
    """ Chunks an Array into several small arrays
    future versions will temporarily save these small array to disc, if needed
    Class is not used yet
    """


    def __init__(self, array_size, chunk_size):
        """ generates an chunked array, where array_size elements takes place and
        is partitioned in chunks of size chunk_size
        """
        self.chunk_size = chunk_size
        self.array_size = array_size
        self.num_chunks = int((array_size - 1) / chunk_size + 1)
        self.data = numpy.zeros((self.num_chunks, chunk_size))

    def get(self, index):
        """ returns array-element of array-index"""
        chunk = int((index + 1) / self.chunk_size)
        chunk_index = int(index % self.chunk_size)
        return self.data[chunk, chunk_index]

    def set(self, index, chunk_object):
        """ returns array-element on array-index"""
        chunk = int((index + 1) / self.chunk_size)
        chunk_index = int(index % self.chunk_size)
        self.data[chunk, chunk_index] = chunk_object

    def get_num_chunks(self):
        """ returns number of chunks used"""
        return self.num_chunks

    def get_chunk_size(self):
        """ returns the number of elements take place in one chunk"""
        return self.chunk_size

    def get_array_size(self):
        """ returns the size of the whole chunked array"""
        return self.array_size

