#!/usr/bin/python

import numpy	
import hmm.kernel.python
import hmm.algorithms
import time as t

# initial conditions
A = numpy.array(
    [[ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ]], dtype=numpy.float32)

B = numpy.array(
    [[ 1.0, 0.0 ],
     [ 0.5, 0.5 ],
     [ 0.0, 1.0 ]], dtype=numpy.float32)

pi = numpy.array([ 0.333, 0.333, 0.333 ], dtype=numpy.float32)

print 'Read data ...'

obs = [
	numpy.loadtxt('data/t1.1000.dat'),
#	numpy.loadtxt('data/t1.2000.dat'),
]
 
maxit = 1
accuracy = 1e-3

kernels = [ hmm.kernel.python ]

print
print 'perform Baum-Welch algorithm'
print 'observation lengths: ', [ len(ob) for ob in obs ]
print 'tested kernel: ', kernels
print 'maximal iterations: ', maxit
print 'accuracy: ', accuracy
for kernel in kernels:
	time_used = 0
	print 'kernel: ', kernel
	for ob in obs:
		print 'observation length: ', len(ob)
		start = t.time()
		A, B, pi, prob, it = \
			hmm.algorithms.baum_welch(
				ob, A, B, pi, 
				maxit=maxit, kernel=kernel, accuracy=accuracy
			)
		print A
		print B
		end = t.time()
		time_used += end-start
		print 'log P( O | A,B,pi ): ', prob
		print 'iterations done: ', maxit
	print 'time used: ', time_used, ' seconds.'
	print

print 'perform Baum-Welch multiple algorithm'
print 'observation lengths: ', [ len(ob) for ob in obs ]
print 'tested kernel: ', kernels
print 'maximal iterations: ', maxit
print 'accuracy: ', accuracy
for kernel in kernels:
	print 'kernel: ', kernel
	start = t.time()
	A, B, pi, prob, it = \
		hmm.algorithms.baum_welch_multiple(
			obs, A, B, pi, 
			maxit=maxit, kernel=kernel, accuracy=accuracy
		)
	print A
	print B 
	end = t.time()
	print 'log P( O | A,B,pi ): ', prob
	print 'iterations done: ', maxit
	print 'time used: ', end-start, ' seconds.'
	print