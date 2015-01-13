import pyopencl
import numpy
import string

def update(A, B, pi, alpha, transitions, states, symbols, probability, N, T):
    kernel.update.update(
            queue,
            (N,N), None,
            A, B, pi, alpha, transitions, states, symbols, probability, numpy.int64(T))

def transition_probabilities(alpha, beta, A, B, ob, T, xi):
    kernel.update.transition_probabilities(
            queue,
            (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
            xi, alpha, beta, A, B, ob, numpy.uint64(T))

def state_probabilities(alpha, beta, T):
    kernel.update.state_probabilities(
            queue,
            (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
            alpha, beta, numpy.uint64(T))

def transition_counts(xi, T, N, scratch):
    # allocate buffer on opencl device
    counts_intermediate = pyopencl.Buffer(
        context, pyopencl.mem_flags.READ_WRITE,
        kernel.WORK_GROUP_SIZE*N*N* numpy.dtype('float32').itemsize)
    counts = pyopencl.Buffer(context, pyopencl.mem_flags.READ_WRITE,
        N*N* numpy.dtype('float32').itemsize)

    # reduce sum to NUM_GROUPS summands
    kernel.update.transition_counts(
            queue, (kernel.WORK_GROUP_SIZE*kernel.WORK_GROUP_SIZE,), (kernel.WORK_GROUP_SIZE,),
            xi, scratch, numpy.uint64(T), counts_intermediate)

    # collect intermedate
    kernel.update.transition_counts(
            queue, (kernel.WORK_GROUP_SIZE,), (kernel.WORK_GROUP_SIZE,),
            counts_intermediate, scratch, numpy.uint64(kernel.WORK_GROUP_SIZE), counts)

    return counts


def symbol_counts(gamma, ob, T, N, M, scratch):
    symbols_intermediate = pyopencl.Buffer(
        context, pyopencl.mem_flags.READ_WRITE,
        kernel.WORK_GROUP_SIZE*N*M* numpy.dtype('float32').itemsize)
    symbols = pyopencl.Buffer(context, pyopencl.mem_flags.READ_WRITE,
        N*M* numpy.dtype('float32').itemsize)

    kernel.update.symbol_counts(
            queue, (kernel.WORK_GROUP_SIZE*kernel.WORK_GROUP_SIZE,), (kernel.WORK_GROUP_SIZE,),
            gamma, ob, scratch, numpy.uint64(T), symbols_intermediate)
    kernel.update.symbol_collect(
            queue, (kernel.WORK_GROUP_SIZE,), (kernel.WORK_GROUP_SIZE,),
            symbols_intermediate, ob, scratch, numpy.uint64(kernel.WORK_GROUP_SIZE), symbols)

    return symbols

def state_counts(gamma, T, N, scratch):
    counts_intermediate = pyopencl.Buffer(
        context, pyopencl.mem_flags.READ_WRITE,
        kernel.WORK_GROUP_SIZE*N* numpy.dtype('float32').itemsize)
    counts = pyopencl.Buffer(
        context, pyopencl.mem_flags.READ_WRITE,
        N* numpy.dtype('float32').itemsize)
    
    kernel.update.state_counts(
        queue, (kernel.WORK_GROUP_SIZE*kernel.WORK_GROUP_SIZE,), (kernel.WORK_GROUP_SIZE,),
        gamma, scratch, numpy.uint64(T), counts_intermediate)
    kernel.update.state_counts(
        queue, (kernel.WORK_GROUP_SIZE,), (kernel.WORK_GROUP_SIZE,),
        counts_intermediate, scratch, numpy.uint64(kernel.WORK_GROUP_SIZE), counts)

    return counts


@profile
def baum_welch(
        sequence,
        transition_probs,
        symbol_probs,
        initial_dist,
        accuracy = 1e-3,
        maxit    = 1):

    A  = pyopencl.Buffer(
            context, 
            pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf=transition_probs)
    B  = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf=symbol_probs)
    pi = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf=initial_dist)
    ob = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
            hostbuf=sequence)

    T = len(sequence)
    N = len(transition_probs)
    M = len(symbol_probs[0])

    alpha = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.READ_WRITE,
            T*N * numpy.dtype('float32').itemsize)

    beta = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.READ_WRITE,
            T*N * numpy.dtype('float32').itemsize)

    matrix_buffer = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.READ_WRITE,
            T*N*N * numpy.dtype('float32').itemsize)

    scratch = pyopencl.LocalMemory(
        kernel.WORK_GROUP_SIZE*N*N* numpy.dtype('float32').itemsize )

    probability = pyopencl.Buffer(
            context,
            pyopencl.mem_flags.WRITE_ONLY,
            numpy.dtype('float32').itemsize )


    old_prob = 0.0
    new_prob = old_prob + accuracy + 1
    it       = 0
    while it < maxit: # abs(new_prob - old_prob) > accuracy and it < maxit:

        forward_naive(ob, A, B, pi, T, N, alpha, matrix_buffer, scratch)
        backward_naive(ob, A, B, T, N, beta, matrix_buffer, scratch)
        transition_probabilities(alpha, beta, A, B, ob, T, matrix_buffer)
        state_probabilities(alpha, beta, T)
        transitions = transition_counts(matrix_buffer, T-1, N, scratch) 
        states = state_counts(alpha, T-1, N, scratch)
        symbols = symbol_counts(alpha, ob, T, N, M, scratch)
        update(A, B, pi, alpha, transitions, states, symbols, probability, N, T)
        if it > 0:
            old_prob = new_prob
        new_prob = numpy.array((1), numpy.float32)
        pyopencl.enqueue_copy(queue, new_prob, probability)
        it = it + 1

    transition_probs = numpy.zeros_like(transition_probs)
    symbol_probs     = numpy.zeros_like(symbol_probs)
    initial_dist     = numpy.zeros_like(initial_dist)

    pyopencl.enqueue_copy(queue, transition_probs, A)
    pyopencl.enqueue_copy(queue, symbol_probs, B)
    pyopencl.enqueue_copy(queue, initial_dist, pi)

    return transition_probs, symbol_probs, initial_dist, new_prob, it



def backward_naive(ob, A, B, T, N, beta, matrices, scratch):

    kernel.backward.build_matrices (
            queue,
            (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
            matrices, beta, A, B, ob, numpy.uint64(T))

    T = T - 1 # matrices does not contain alpha_0!

    # forward upwind method
    last_results = matrices
    stack = [ (last_results, T) ]
    while T > 1:
        grouped_results = pyopencl.Buffer(context, 
            pyopencl.mem_flags.READ_WRITE, (T/kernel.WORK_GROUP_SIZE+1)*N*N*4)
        kernel.backward.scan (
                queue,
                (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
                grouped_results, last_results, scratch, numpy.uint64(T))
        T = T / kernel.WORK_GROUP_SIZE
        stack.append( (grouped_results, T) )
        last_results = grouped_results

    # forward rewind method
    grouped_results = last_results
    while stack:
        last_results, T = stack.pop()
        kernel.backward.propagate (
                queue,
                (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
                last_results, grouped_results, numpy.uint64(T))
        grouped_results = last_results

    T = T + 1
    kernel.backward.multiply_with_beta_T(
                queue,
                (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
                beta, last_results, numpy.uint64(T))

    return beta


def forward_naive(ob, A, B, pi, T, N, alpha, matrices, scratch):

    kernel.forward.build_matrices (
            queue,
            (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
            matrices, alpha, A, B, pi, ob, numpy.uint64(T))

    T = T - 1 # matrices does not contain alpha_0!

    # forward upwind method
    last_results = matrices
    stack = [ (last_results, T) ]
    while T > 1:
        grouped_results = pyopencl.Buffer(context, 
            pyopencl.mem_flags.READ_WRITE, (T/kernel.WORK_GROUP_SIZE+1)*N*N*4)
        kernel.forward.scan (
                queue,
                (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
                grouped_results, last_results, scratch, numpy.uint64(T))
        T = T / kernel.WORK_GROUP_SIZE
        stack.append( (grouped_results, T) )
        last_results = grouped_results

    # forward rewind method
    grouped_results = last_results
    while stack:
        last_results, T = stack.pop()
        kernel.forward.propagate (
                queue,
                (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
                last_results, grouped_results, numpy.uint64(T))
        grouped_results = last_results

    T = T + 1
    kernel.forward.multiply_with_alpha_0(
                queue,
                (kernel.NUM_UNITS,), (kernel.WORK_GROUP_SIZE,),
                alpha, last_results, numpy.uint64(T))

    return alpha


_FORWARD_SOURCE_SCALING  = 'lib/opencl/forward_naive_scaling.cl'
_BACKWARD_SOURCE_SCALING = 'lib/opencl/backward_naive_scaling.cl'
_UPDATE_SOURCE           = 'lib/opencl/update.cl'

platform = pyopencl.get_platforms()[0]
device   = platform.get_devices()[0]
context  = pyopencl.Context([device])
queue    = pyopencl.CommandQueue(context)

class Kernel:
    WORK_GROUP_SIZE = 2
    NUM_UNITS = 2*2
    NUM_GROUPS = 2

    def __init__(self, N=3, M=2):
        self.compile(N,M)

    def compile(self, N, M):
        ff = open(_FORWARD_SOURCE_SCALING, 'r')
        fb = open(_BACKWARD_SOURCE_SCALING, 'r')
        upd = open(_UPDATE_SOURCE, 'r')

        source = string.Template("".join(ff.readlines())).substitute(
            N=N, M=M, precision="float")
        self.forward = pyopencl.Program(context, source).build()

        source = string.Template("".join(fb.readlines())).substitute(
            N=N, M=M, precision="float")
        self.backward = pyopencl.Program(context, source).build()

        source = string.Template("".join(upd.readlines())).substitute(
            N=N, M=M, precision="float") 
        self.update = pyopencl.Program(context, source).build()

kernel = Kernel()