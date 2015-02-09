import spscicomp.hmm.kernel.opencl
import spscicomp.hmm.kernel.fortran
import spscicomp.hmm.kernel.python
import spscicomp.hmm.kernel.c
import spscicomp.hmm.algorithms
import spscicomp.hmm.utility
import spscicomp.hmm.lib.c
import numpy
import pyopencl
import time as t

N, M = 3, 2


transition_matrix, symbol_probablity, initial_distribution = spscicomp.hmm.utility.get_models()['t2']

numpy.set_printoptions(precision=3, suppress=True)

print transition_matrix
print symbol_probablity
print initial_distribution

print 'Reading data'

obs = [
	numpy.loadtxt('data/hmm1.3000000.dat', dtype=numpy.int16),
#	numpy.array([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=numpy.int16)
]

maxit = 100

print 'start opencl'
start = t.time()
print spscicomp.hmm.kernel.opencl.baum_welch(
		obs[0],
		transition_matrix,
		symbol_probablity,
		initial_distribution,
		maxit=maxit, accuracy=-1)
end = t.time()
print 'opencl baum welch', end-start


print 'start c'
start = t.time()
print spscicomp.hmm.algorithms.baum_welch_multiple([obs[0]], transition_matrix,
	symbol_probablity, initial_distribution,
	maxit=maxit, kernel=spscicomp.hmm.kernel.c, accuracy=-1, dtype=numpy.float32)
end = t.time()
print 'c baum welch:', end-start


# print 'start fortran'
# start = t.time()
# print hmm.algorithms.baum_welch_multiple([obs[0]], transition_matrix,
# 	symbol_probablity, initial_distribution,
# 	maxit=maxit, kernel=hmm.kernel.fortran, accuracy=-1, dtype=numpy.float32)	
# end = t.time()
# print 'fortran baum welch:', end-start

