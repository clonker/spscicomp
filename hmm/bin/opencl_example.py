import hmm.kernel.opencl as hmmcl
import numpy
import pyopencl
import time as t

N, M = 3, 2

context, queue = hmmcl.initialize(N, M)

transition_matrix = numpy.array(
	[[ 0.333, 0.333, 0.333],
	 [ 0.333, 0.333, 0.333],
	 [ 0.333, 0.333, 0.333]], dtype=numpy.float32
)

symbol_probablity = numpy.array(
	[[ 0.5, 0.5],
	 [ 1.0, 0.0],
	 [ 0.0, 1.0]], dtype=numpy.float32
)

initial_distribution = numpy.array(
	[ 0.333, 0.333, 0.333 ], dtype=numpy.float32
)

print 'Reading data'
observation = numpy.loadtxt('data/t1.1000000.dat', dtype=numpy.int16)


A = pyopencl.Buffer(
		context,
		pyopencl.mem_flags.READ_ONLY |
		pyopencl.mem_flags.COPY_HOST_PTR,
		hostbuf=transition_matrix
	)

B = pyopencl.Buffer(
		context,
		pyopencl.mem_flags.READ_ONLY |
		pyopencl.mem_flags.COPY_HOST_PTR,
		hostbuf=symbol_probablity
	)

pi = pyopencl.Buffer(
		context,
		pyopencl.mem_flags.READ_ONLY |
		pyopencl.mem_flags.COPY_HOST_PTR,
		hostbuf=initial_distribution
	)

ob = pyopencl.Buffer(
		context,
		pyopencl.mem_flags.READ_ONLY |
		pyopencl.mem_flags.COPY_HOST_PTR,
		hostbuf=observation
	)

T = len(observation)

print 'Start forward'

start = t.time()
for i in range(10):
	alpha_buffer, _, _ = hmmcl.forward(A, B, pi, ob, T, num_groups=T/256, num_units=64)
end = t.time()

print end-start