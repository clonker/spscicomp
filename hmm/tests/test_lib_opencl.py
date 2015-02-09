import unittest
import pyopencl
import string
import numpy
import spscicomp.hmm.utility
import spscicomp.hmm.kernel.python as pyhmm
import spscicomp.hmm.kernel.c as chmm
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

    # def test_forward_all(self):
    #     A, B, pi = hmm.utility.get_models()['t2']
    #     ob = numpy.loadtxt('data/hmm1.3000000.dat', numpy.int16)
    #     T = len(ob)
    #     N,M = B.shape
    #     kernel = self.get_kernel(N, M, T)
    #     mf = pyopencl.mem_flags

    #     # prepare buffers
    #     bA = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    #     bB = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    #     bpi = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
    #     bob = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
    #     bC = pyopencl.Buffer(context, mf.READ_WRITE,
    #         numpy.dtype('float32').itemsize * T*N*N)
    #     balpha = pyopencl.Buffer(context, mf.READ_WRITE,
    #         numpy.dtype('float32').itemsize * T*N)

    #     start = t.time()
    #     _, alpha_ns = chmm.forward_no_scaling(A, B, pi, ob)
    #     end = t.time()
    #     c_time = end-start
    #     print 'forward no scaling time in c: {t} seconds.'.format(t=c_time)

    #     start = t.time()
    #     _, alpha_ref, _ = chmm.forward(A, B, pi, ob)
    #     end = t.time()
    #     c_time = end-start
    #     print 'forward time in c: {t} seconds.'.format(t=c_time)

    #     blockss    = [2*n for n in range(1,150)]
    #     blocksizes = [2**n for n in range(9)]

    #     minimum = (0, 0, 1.0)
    #     print 'benchmarking hmm.lib.opencl.forward.scan_all'
    #     for blocksize in blocksizes:
    #         for blocks in blockss:
    #             if blocks <= 2*blocksize and blocks*blocksize*2 <= T:
    #                 TOTAL_UNITS = 2*blocksize*blocks
    #                 BLOCK_UNITS = 2*blocksize

    #                 breduced = pyopencl.Buffer(context, mf.READ_WRITE,
    #                     numpy.dtype('float32').itemsize * blocks*N*N)
    #                 scratch = pyopencl.LocalMemory(
    #                     numpy.dtype('float32').itemsize*BLOCK_UNITS*N*N)
    #                 transform = numpy.array(
    #                     calculate_index_table([0], numpy.log2(blocksize)+1),
    #                     numpy.int32)
    #                 btransform = pyopencl.Buffer(context,
    #                     mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=transform)

    #                 e = pyopencl.enqueue_barrier(queue)
    #                 e.wait()
    #                 start = t.time()
    #                 kernel.initialize(queue,
    #                         (64*256,), (256,),
    #                         bC, balpha, bA, bB, bpi, bob, numpy.int32(T))
    #                 kernel.reduce(queue,
    #                         (blocks*blocksize,), (blocksize,),
    #                         bC, breduced, scratch, btransform, numpy.int32(T))
    #                 kernel.scan_toplevel(queue,
    #                         (blocksize,), (blocksize,),
    #                         breduced, scratch, btransform, numpy.int32(blocks))
    #                 kernel.scan_all(queue,
    #                         (blocks*blocksize,), (blocksize,),
    #                         bC, breduced, scratch, btransform, numpy.int32(T))
    #                 kernel.finalize(queue,
    #                         (64*256,), (256,),
    #                         bC, balpha, numpy.int32(T))
    #                 e = pyopencl.enqueue_barrier(queue)
    #                 e.wait()
    #                 end = t.time()

    #                 alpha = numpy.zeros((T,N), numpy.float32)
    #                 pyopencl.enqueue_copy(queue, alpha, balpha)
    #                 # numpy.testing.assert_almost_equal(alpha, alpha_ref, 1)

    #                 if end-start <= minimum[2] and \
    #                    blocks    >= minimum[1] and \
    #                    blocksize >= minimum[0]:
    #                    minimum = (blocksize, blocks, end-start)

    #     print "minimum of {t} seconds achieved with block size {bs} and {b} groups".format(
    #         b=minimum[1], bs=minimum[0], t=minimum[2])
    #     print "speedup: {s}".format(s=c_time/(minimum[2]))

    # def test_forward_scan_bottom_level(self):
    #     A, B, pi = hmm.utility.get_models()['t2']
    #     ob = numpy.loadtxt('data/hmm1.3000000.dat', numpy.int16)
    #     T = len(ob)
    #     N,M = B.shape
    #     kernel = self.get_kernel(N, M, T)
    #     mf = pyopencl.mem_flags

    #     # prepare buffers
    #     bA = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    #     bB = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    #     bpi = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
    #     bob = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
    #     bC = pyopencl.Buffer(context, mf.READ_WRITE,
    #         numpy.dtype('float32').itemsize * T*N*N)
    #     balpha_T = pyopencl.Buffer(context, mf.READ_WRITE,
    #         numpy.dtype('float32').itemsize * N*N)

    #     start = t.time()
    #     _, alpha_ns = chmm.forward_no_scaling(A, B, pi, ob)
    #     end = t.time()
    #     c_time = end-start
    #     print 'forward no scaling time in c: {t} seconds.'.format(t=c_time)

    #     start = t.time()
    #     _, alpha, _ = chmm.forward(A, B, pi, ob)
    #     end = t.time()
    #     c_time = end-start
    #     print 'forward time in c: {t} seconds.'.format(t=c_time)

    #     blockss    = [2*n for n in range(1,150)]
    #     blocksizes = [2**n for n in range(9)]

    #     minimum = (0, 0, 1.0)
    #     print 'benchmarking hmm.lib.opencl.forward.scan_all'
    #     for blocksize in blocksizes:
    #         for blocks in blockss:
    #             if blocks <= 2*blocksize and blocks*blocksize*2 <= T:
    #                 TOTAL_UNITS = 2*blocksize*blocks
    #                 BLOCK_UNITS = 2*blocksize

    #                 balpha = pyopencl.Buffer(context, mf.READ_WRITE,
    #                     numpy.dtype('float32').itemsize * blocks*N*N)

    #                 scratch = pyopencl.LocalMemory(
    #                     numpy.dtype('float32').itemsize*BLOCK_UNITS*N*N)

    #                 transform = numpy.array(
    #                     calculate_index_table([0], numpy.log2(blocksize)+1),
    #                     numpy.int32)
    #                 btransform = pyopencl.Buffer(context,
    #                     mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=transform)

    #                 e = pyopencl.enqueue_barrier(queue)
    #                 e.wait()
    #                 start = t.time()
    #                 kernel.build_matrices(queue,
    #                         (64*256,), (256,),
    #                         bC, bA, bB, bpi, bob, numpy.int32(T))
    #                 kernel.reduce(queue,
    #                         (blocks*blocksize,), (blocksize,),
    #                         bC, balpha, scratch, btransform, numpy.int32(T))
    #                 kernel.scan_toplevel(queue,
    #                         (blocksize,), (blocksize,),
    #                         balpha, scratch, btransform, numpy.int32(blocks))
    #                 kernel.scan_all(queue,
    #                         (blocks*blocksize,), (blocksize,),
    #                         bC, balpha, scratch, btransform, numpy.int32(T))
    #                 e = pyopencl.enqueue_barrier(queue)
    #                 e.wait()
    #                 end = t.time()

    #                 # C_ref = forward_build_matrices(ob, A, B, pi)
    #                 # C_ref[0] = C_ref[0] / numpy.sum(C_ref[0])
    #                 # for i in range(1,len(C)):
    #                 #     C_ref[i] = C_ref[i].dot(C_ref[i-1])
    #                 #     C_ref[i] = C_ref[i] / numpy.sum(C_ref[i])
    #                 # print 'C should be:\n', C_ref
    #                 # numpy.testing.assert_almost_equal(C, C_ref, 5)

    #                 if end-start <= minimum[2] and \
    #                    blocks    >= minimum[1] and \
    #                    blocksize >= minimum[0]:
    #                    minimum = (blocksize, blocks, end-start)

    #     print "minimum of {t} seconds achieved with block size {bs} and {b} groups".format(
    #         b=minimum[1], bs=minimum[0], t=minimum[2])
    #     print "speedup: {s}".format(s=c_time/(minimum[2]))

    # def test_forward_scan_top_level(self):
    #     A, B, pi = hmm.utility.get_models()['t2']
    #     ob = numpy.loadtxt('data/hmm1.1000.dat', numpy.int16)
    #     T = len(ob)
    #     N,M = B.shape
    #     kernel = self.get_kernel(N, M, T)
    #     mf = pyopencl.mem_flags

    #     # prepare buffers
    #     bA = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    #     bB = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    #     bpi = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
    #     bob = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
    #     bC = pyopencl.Buffer(context, mf.READ_WRITE,
    #         numpy.dtype('float32').itemsize * T*N*N)
    #     balpha_T = pyopencl.Buffer(context, mf.WRITE_ONLY,
    #         numpy.dtype('float32').itemsize * N*N)

    #     kernel.build_matrices(queue,
    #             (64*256,), (256,),
    #             bC, bA, bB, bpi, bob, numpy.int32(T))

    #     start = t.time()
    #     _, alpha, _ = chmm.forward(A, B, pi, ob)
    #     end = t.time()
    #     c_time = end-start
    #     print 'forward time in c: {t} seconds.'.format(t=c_time)

    #     blockss    = [7*n for n in range(1,40)]
    #     blocksizes = [2**n for n in range(9)]

    #     minimum = (0, 0, 1.0)
    #     print 'benchmarking hmm.lib.opencl.forward.scan_top_level'
    #     for blocksize in blocksizes:
    #         for blocks in blockss:
    #             if blocks <= 2*blocksize and blocks*blocksize*2 <= T:
    #                 TOTAL_UNITS = 2*blocksize*blocks
    #                 BLOCK_UNITS = 2*blocksize

    #                 balpha = pyopencl.Buffer(context, mf.READ_WRITE,
    #                     numpy.dtype('float32').itemsize * blocks*N*N)
    #                 scratch = pyopencl.LocalMemory(
    #                     numpy.dtype('float32').itemsize*BLOCK_UNITS*N*N)

    #                 transform = numpy.array(
    #                     calculate_index_table([0], numpy.log2(blocksize)+1),
    #                     numpy.int32)
    #                 btransform = pyopencl.Buffer(context,
    #                     mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=transform)

    #                 e = pyopencl.enqueue_barrier(queue)
    #                 e.wait()

    #                 start = t.time()
    #                 kernel.reduce(queue,
    #                         (blocks*blocksize,), (blocksize,),
    #                         bC, balpha, scratch, btransform, numpy.int32(T))
    #                 kernel.scan_toplevel(queue,
    #                         (blocksize,), (blocksize,),
    #                         balpha, scratch, btransform, numpy.int32(blocks))
    #                 e = pyopencl.enqueue_barrier(queue)
    #                 e.wait()
    #                 end = t.time()

    #                 C = numpy.zeros((T,N,N), numpy.float32)
    #                 pyopencl.enqueue_copy(queue, C, bC)
    #                 scanned = numpy.zeros((blocks,N,N), numpy.float32)
    #                 pyopencl.enqueue_copy(queue, scanned, balpha)

    #                 C_T = scanned[blocks-1]
    #                 T0 = (T / BLOCK_UNITS) * BLOCK_UNITS
    #                 for k in range(T%BLOCK_UNITS):
    #                     C_T = C[T0+k].dot(C_T)
    #                     C_T = C_T / numpy.sum(C_T)
    #                 alpha0 = pi * B[:, ob[0]]
    #                 alphaT = C_T.dot(alpha0)
    #                 alphaT = alphaT / numpy.sum(alphaT)
    #                 numpy.testing.assert_almost_equal(alphaT, alpha[T-1])

    #                 if end-start <= minimum[2] and \
    #                    blocks    >= minimum[1] and \
    #                    blocksize >= minimum[0]:
    #                    minimum = (blocksize, blocks, end-start)

    #     print "minimum of {t} seconds achieved with block size {bs} and {b} groups".format(
    #         b=minimum[1], bs=minimum[0], t=minimum[2])
    #     print "speedup: {s}".format(s=c_time/(minimum[2]))


    # def test_forward_reduce(self):
    #     A, B, pi = hmm.utility.get_models()['t2']
    #     ob = numpy.loadtxt('data/hmm1.1000.dat', numpy.int16) # numpy.array([1,0,1,0,1], numpy.int16)
    #     T = len(ob)
    #     N,M = B.shape
    #     kernel = self.get_kernel(N, M, T)
    #     mf = pyopencl.mem_flags

    #     # prepare buffers
    #     bA = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    #     bB = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    #     bpi = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
    #     bob = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
    #     bC = pyopencl.Buffer(context, mf.READ_WRITE,
    #         numpy.dtype('float32').itemsize * T*N*N)
    #     balpha_T = pyopencl.Buffer(context, mf.WRITE_ONLY,
    #         numpy.dtype('float32').itemsize * N*N)

    #     # execute kernels
    #     kernel.build_matrices(queue,
    #             (64*256,), (256,),
    #             bC, bA, bB, bpi, bob, numpy.int32(T))
    #     e = pyopencl.enqueue_barrier(queue)
    #     e.wait()

    #     start = t.time()
    #     _, alpha, _ = chmm.forward(A, B, pi, ob)
    #     end = t.time()
    #     c_time = end-start
    #     print 'forward time in c: {t} seconds.'.format(t=c_time)

    #     blockss    = [14*n for n in range(1,20)]
    #     blocksizes = [2**n for n in range(9)]

    #     minimum = (0, 0, 1.0)
    #     print 'benchmarking hmm.lib.opencl.forward.reduce'
    #     for blocksize in blocksizes:
    #         for blocks in blockss:
    #             if blocks <= 2*blocksize and blocks*blocksize*2 <= T:
    #                 TOTAL_UNITS = 2*blocksize*blocks
    #                 BLOCK_UNITS = 2*blocksize
    #                 balpha = pyopencl.Buffer(context, mf.READ_WRITE,
    #                     numpy.dtype('float32').itemsize * blocks*N*N)
    #                 scratch = pyopencl.LocalMemory(
    #                     numpy.dtype('float32').itemsize*BLOCK_UNITS*N*N)
    #                 transform = numpy.array(
    #                     calculate_index_table([0], numpy.log2(blocksize)+1),
    #                     numpy.int32)
    #                 btransform = pyopencl.Buffer(context,
    #                     mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=transform)
    #                 start = t.time()
    #                 kernel.reduce(queue,
    #                         (blocks*blocksize,), (blocksize,),
    #                         bC, balpha, scratch, btransform, numpy.int32(T))
    #                 e = pyopencl.enqueue_barrier(queue)
    #                 e.wait()
    #                 end = t.time()
    #                 C = numpy.zeros((T,N,N), numpy.float32)
    #                 pyopencl.enqueue_copy(queue, C, bC)
    #                 reduced = numpy.zeros((blocks,N,N), numpy.float32)
    #                 pyopencl.enqueue_copy(queue, reduced, balpha)
    #                 C_T = reduced[0]
    #                 for k in range(1, len(reduced)):
    #                     C_T = reduced[k].dot(C_T)
    #                     C_T = C_T / numpy.sum(C_T)
    #                 T0 = (T/BLOCK_UNITS) * BLOCK_UNITS
    #                 for k in range(T%BLOCK_UNITS):
    #                     C_T = C[T0+k].dot(C_T)
    #                     C_T = C_T / numpy.sum(C_T)
    #                 alpha0 = pi * B[:, ob[0]]
    #                 alphaT = C_T.dot(alpha0)
    #                 alphaT = alphaT / numpy.sum(alphaT)
    #                 numpy.testing.assert_almost_equal(alphaT, alpha[T-1])

    #                 if end-start <= minimum[2] and \
    #                    blocks    >= minimum[1] and \
    #                    blocksize >= minimum[0]:
    #                    minimum = (blocksize, blocks, end-start)

    #     print "minimum of {t} seconds achieved with block size {bs} and {b} groups".format(
    #         b=minimum[1], bs=minimum[0], t=minimum[2])
    #     print "speedup: {s}".format(s=c_time/(minimum[2]))


    # def test_forward_build_matrices(self):
    #     A, B, pi = hmm.utility.get_models()['t2']
    #     ob = numpy.loadtxt('data/hmm1.10.dat', numpy.int16)
    #     T = len(ob)
    #     N,M = B.shape
    #     kernel = self.get_kernel(N, M, T)
    #     mf = pyopencl.mem_flags

    #     # prepare buffers
    #     bA = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    #     bB = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    #     bpi = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
    #     bob = pyopencl.Buffer(context,
    #         mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
    #     bC = pyopencl.Buffer(context, mf.WRITE_ONLY,
    #         numpy.dtype('float32').itemsize * T*N*N)

    #     # set up many different work group sizes
    #     block_sizes = [2**n for n in range(9)]
    #     num_groupss = [14*n for n in range(1,9)]

    #     print 'benchmarking hmm.lib.opencl.forward.build_matrices'
    #     minimum = (0, 0, 1.0)

    #     T = numpy.int32(T)

    #     C_ref = forward_build_matrices(ob, A, B, pi)
    #     for block_size in block_sizes:
    #         for num_groups in num_groupss:
    #             C = numpy.zeros((T, N, N), numpy.float32)
    #             start = t.time()
    #             kernel.build_matrices(queue,
    #                 (block_size*num_groups,), (block_size,),
    #                 bC, bA, bB, bpi, bob, T)
    #             e = pyopencl.enqueue_barrier(queue)
    #             e.wait()
    #             end = t.time()
    #             pyopencl.enqueue_copy(queue, C, bC)
    #             if block_size >= minimum[0] and \
    #                num_groups >= minimum[1] and \
    #                end-start <= minimum[2]:
    #                 minimum = (block_size, num_groups, end-start)
    #             numpy.testing.assert_almost_equal(C, C_ref)
    #     print "minimum of {t} seconds achieved with " \
    #           "block size {b} and {g} groups".format(
    #             t=minimum[2], b=minimum[0], g=minimum[1])
                

class TestSameAsMatrixMult(unittest.TestCase):

    def test_forward_same_as_matrix_mult(self):
        A, B, pi = spscicomp.hmm.utility.get_models()['t2']
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
