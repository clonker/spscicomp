import hmm.kernel.opencl as hmmcl
import hmm.kernel.fortran
import hmm.kernel.python
import hmm.lib.c
import numpy
import pyopencl
import time as t

N, M = 3, 2


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

ctx = hmmcl.Context(3, 2)

print 'Reading data'
observation = numpy.loadtxt('data/t1.3000000.dat', dtype=numpy.int16)
observation2 = numpy.array([1, 0, 1, 0, 1], dtype=numpy.int16)

T = len(observation)

# beta = hmmcl.backward_naive(ctx, transition_matrix,
# 	symbol_probablity, initial_distribution, observation2)
# beta_ = numpy.zeros((T,N), dtype=numpy.float32)
# pyopencl.enqueue_copy(ctx.queue, beta_, beta)
# print beta_

# alpha = hmmcl.forward_no_scaling_naive(ctx, transition_matrix,
# 	symbol_probablity, initial_distribution, observation2)
# alpha_ = numpy.zeros((T,N), dtype=numpy.float32)
# pyopencl.enqueue_copy(ctx.queue, alpha_, alpha)
# print alpha_


# _, alpha_ = hmm.kernel.python.forward_no_scaling(
# 	transition_matrix, symbol_probablity, initial_distribution, observation2)
# print alpha_

start = t.time()
for i in range(10):
	hmmcl.forward_no_scaling_naive(ctx, transition_matrix,
		symbol_probablity, initial_distribution, observation)
	event = pyopencl.enqueue_barrier(ctx.queue)
	event.wait()
end = t.time()

print 'opencl forward no scaling naive:', end-start


start = t.time()
for i in range(10):
	hmmcl.forward_naive(ctx, transition_matrix,
		symbol_probablity, initial_distribution, observation)
	event = pyopencl.enqueue_barrier(ctx.queue)
	event.wait()
end = t.time()

print 'opencl forward scaling naive:', end-start

start = t.time()
for i in range(10):
	hmm.kernel.fortran.forward(transition_matrix, symbol_probablity, initial_distribution, observation)
end = t.time()

print 'fortran forward scaling:', end-start

start = t.time()
for i in range(10):
	hmm.lib.c.forward32(transition_matrix, symbol_probablity, initial_distribution, observation)
end = t.time()

print 'c forward scaling:', end-start