import unittest
import pyopencl
import string
import numpy
import hmm.utility
import hmm.kernel.python as pyhmm
import hmm.kernel.c as chmm
import time as t

platform = pyopencl.get_platforms()[0]
device   = platform.get_devices()[0]
context  = pyopencl.Context([device])
queue    = pyopencl.CommandQueue(context)

numpy.set_printoptions(precision=3)

def calculate_index_table(table, depth):
    if depth == 0:
        return table
    else:
        table = [2*t for t in table]
        table = table + [t+1 for t in table]
        return calculate_index_table(table, depth-1)

def get_kernel(N, M, T, precision):
    fd = open(SOURCE, 'r')
    source = string.Template("".join(fd.readLines()))
    source = source.substitute(N=N, M=M, T=T, precision=precision)
    return pyopencl.Program(context, source).build()

def forward_build_matrices(ob, A, B, pi):
    T = len(ob)
    N, M = B.shape
    C = numpy.zeros((T,N,N))
    C[0] = numpy.eye(N)
    for t in range(1,T):
        for i in range(N):
            for j in range(N):
                C[t,i,j] = B[i,ob[t]]*A[j,i]
    return C

class TestTemplateKernel(unittest.TestCase):

    SOURCE = ''

    def get_kernel(self, N, M, T, precision = 'float'):
        fd = open(self.SOURCE, 'r')
        source = string.Template("".join(fd.readlines()))
        source = source.substitute(N=N, M=M, T=T, precision=precision)
        return pyopencl.Program(context, source).build()

class TestOpenCLMatchesPython(TestTemplateKernel):

    def setUp(self):
        self.SOURCE = 'lib/opencl/forward.cl'

    def test_forward_reduce(self):
        A, B, pi = hmm.utility.get_models()['t2']
        ob = numpy.loadtxt('data/hmm1.1000000.dat', numpy.int16) # numpy.array([1,0,1,0], numpy.int16)
        T = len(ob)
        N,M = B.shape
        kernel = self.get_kernel(N, M, T)
        mf = pyopencl.mem_flags

        # prepare buffers
        bA = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        bB = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        bpi = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
        bob = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
        bC = pyopencl.Buffer(context, mf.READ_WRITE,
            numpy.dtype('float32').itemsize * T*N*N)
        balpha_T = pyopencl.Buffer(context, mf.WRITE_ONLY,
            numpy.dtype('float32').itemsize * N*N)

        start = t.time()
        _, alpha, _ = chmm.forward(A, B, pi, ob)
        end = t.time()
        c_time = end-start
        print 'forward time in c: {t} seconds.'.format(t=c_time)
        T = numpy.int32(T)

        blockss    = [14*n for n in range(1,40)]
        blocksizes = [2**n for n in range(9)]

        minimum = (0, 0, 1.0)
        print 'benchmarking hmm.lib.opencl.forward.reduce'
        for blocksize in blocksizes:
            for blocks in blockss:
                if blocks < 2*blocksize:
                    balpha = pyopencl.Buffer(context, mf.READ_WRITE,
                        numpy.dtype('float32').itemsize * blocks*N*N)
                    scratch = pyopencl.LocalMemory(
                        numpy.dtype('float32').itemsize*2*blocksize*N*N)

                    transform = numpy.array(
                        calculate_index_table([0], numpy.log2(blocksize)+1),
                        numpy.int32)
                    btransform = pyopencl.Buffer(context,
                        mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=transform)

                    # execute kernels
                    kernel.build_matrices(queue,
                                (64*256,), (256,),
                                bC, bA, bB, bpi, bob, T)

                    start = t.time()
                    kernel.reduce(queue,
                                (blocks*blocksize,), (blocksize,),
                                bC, balpha, scratch, btransform, T)
                    kernel.reduce(queue,
                                (blocksize,), (blocksize,),
                                balpha, balpha_T, scratch, btransform, numpy.int32(blocks))
                    e = pyopencl.enqueue_barrier(queue)
                    e.wait()
                    end = t.time()

                    C_T = numpy.zeros((N,N), numpy.float32)
                    pyopencl.enqueue_copy(queue, C_T, balpha_T)
                    alpha0 = pi * B[:, ob[0]]
                    alphaT = C_T.dot(alpha0)
                    alphaT = alphaT / numpy.sum(alphaT)
                    numpy.testing.assert_almost_equal(alphaT, alpha[T-1])

                    if end-start <= minimum[2] and \
                       blocks    >= minimum[1] and \
                       blocksize >= minimum[0]:
                       minimum = (blocksize, blocks, end-start)

        print "minimum of {t} seconds achieved with block size {bs} and {b} groups".format(
            b=minimum[1], bs=minimum[0], t=minimum[2])
        print "speedup: {s}".format(s=c_time/(minimum[2]))


        # C = forward_build_matrices(ob, A, B, pi)
        # C_T = numpy.eye(N)
        # for C_t in C:
        #     C_T = C_t.dot(C_T)
        # print C_T


    def test_forward_build_matrices(self):
        A, B, pi = hmm.utility.get_models()['t2']
        ob = numpy.loadtxt('data/hmm1.1000.dat', numpy.int16)
        T = len(ob)
        N,M = B.shape
        kernel = self.get_kernel(N, M, T)
        mf = pyopencl.mem_flags

        # prepare buffers
        bA = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        bB = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        bpi = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
        bob = pyopencl.Buffer(context,
            mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
        bC = pyopencl.Buffer(context, mf.WRITE_ONLY,
            numpy.dtype('float32').itemsize * T*N*N)

        # set up many different work group sizes
        block_sizes = [2**n for n in range(9)]
        num_groupss = [14*n for n in range(1,9)]

        print 'benchmarking hmm.lib.opencl.forward.build_matrices'
        minimum = (0, 0, 1.0)

        T = numpy.int32(T)

        C_ref = forward_build_matrices(ob, A, B, pi)
        for block_size in block_sizes:
            for num_groups in num_groupss:
                C = numpy.zeros((T, N, N), numpy.float32)
                start = t.time()
                kernel.build_matrices(queue,
                    (block_size*num_groups,), (block_size,),
                    bC, bA, bB, bpi, bob, T)
                e = pyopencl.enqueue_barrier(queue)
                e.wait()
                end = t.time()
                pyopencl.enqueue_copy(queue, C, bC)
                if block_size >= minimum[0] and \
                   num_groups >= minimum[1] and \
                   end-start <= minimum[2]:
                    minimum = (block_size, num_groups, end-start)
                numpy.testing.assert_almost_equal(C, C_ref)
        print "minimum of {t} seconds achieved with " \
              "block size {b} and {g} groups".format(
                t=minimum[2], b=minimum[0], g=minimum[1])
                

class TestSameAsMatrixMult(unittest.TestCase):

    def test_forward_same_as_matrix_mult(self):
        A, B, pi = hmm.utility.get_models()['t2']
        ob = numpy.array([1, 0, 1, 0, 1], numpy.int16)
        T = len(ob)
        N,M = B.shape

        C = forward_build_matrices(ob, A, B, pi)
        alpha = numpy.zeros((T,N))
        alpha[0] = pi * B[:, ob[0]]

        for t in range(1,T):
            alpha[t] = C[t].dot(alpha[t-1])

        _, alpha_ref = pyhmm.forward_no_scaling(A, B, pi, ob)
        
        numpy.testing.assert_almost_equal(alpha, alpha_ref)
