import hmm.kernel.opencl
import hmm.kernel.fortran
import hmm.kernel.python
import hmm.kernel.c
import hmm.algorithms
import hmm.utility
import hmm.lib.c
import numpy
import pyopencl
import time as t

N, M = 3, 2


transition_matrix, symbol_probablity, initial_distribution = hmm.utility.get_models()['t2']

numpy.set_printoptions(precision=3)

print transition_matrix
print symbol_probablity
print initial_distribution

print 'Reading data'

obs = [
	# numpy.loadtxt('data/t1.100.dat', dtype=numpy.int16),
	numpy.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1], dtype=numpy.int16)
]

maxit = 1


start = t.time()
print hmm.algorithms.baum_welch(obs[0], transition_matrix,
	symbol_probablity, initial_distribution, maxit=maxit, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32)
end = t.time()
print 'c baum welch:', end-start


start = t.time()
print hmm.kernel.opencl.baum_welch(
		obs[0],
		transition_matrix,
		symbol_probablity,
		initial_distribution,
		maxit=maxit, accuracy=-1)
end = t.time()

print 'opencl baum welch', end-start

# alpha = numpy.zeros((T,N), numpy.float32)
# pyopencl.enqueue_copy(hmm.kernel.opencl.queue, alpha, alpha_cl)
# beta = numpy.zeros((T,N), numpy.float32)
# pyopencl.enqueue_copy(hmm.kernel.opencl.queue, beta, beta_cl)
# print 'alpha: '
# print alpha
# print 'beta: '
# print beta

# transitions = numpy.zeros((N,N), numpy.float32)
# pyopencl.enqueue_copy(hmm.kernel.opencl.queue, transitions, transitions_cl)
# states = numpy.zeros((N), numpy.float32)
# pyopencl.enqueue_copy(hmm.kernel.opencl.queue, states, states_cl)
# symbols = numpy.zeros((N,M), numpy.float32)
# pyopencl.enqueue_copy(hmm.kernel.opencl.queue, symbols, symbols_cl)
# print 'transitions: '
# print transitions
# print 'symbols: '
# print symbols
# print 'states: '
# print states

# _, alpha = hmm.kernel.python.forward_no_scaling(transition_matrix,
# 	symbol_probablity, initial_distribution, observation2)
# print 'alpha python: '
# print alpha

# beta = hmm.kernel.python.backward_no_scaling(transition_matrix,
# 	symbol_probablity, observation2)
# print 'beta python: '
# print beta

# gamma = hmm.kernel.python.state_probabilities(alpha, beta)
# print 'gamma python:\n', gamma

# xi = hmm.kernel.python.transition_counts(alpha, beta,
# 	transition_matrix, symbol_probablity, observation2)

# print 'xi python:\n',xi

# gamma_counts = hmm.kernel.python.state_counts(gamma, T-1)
# print 'gamma counts python:\n', gamma_counts


# print 'opencl baum_welch:', end-start


# start = t.time()
# for i in range(maxit):
# 	hmm.kernel.fortran.forward(transition_matrix, symbol_probablity, initial_distribution, observation)
# end = t.time()

# print 'fortran forward scaling:', end-start

# start = t.time()
# for i in range(maxit):
# 	hmm.lib.c.forward32(transition_matrix, symbol_probablity, initial_distribution, observation)
# end = t.time()

# print 'c forward scaling:', end-start
