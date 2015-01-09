#!/usr/bin/python

import numpy    
import hmm.kernel.c
import hmm.kernel.python
import hmm.kernel.fortran
import hmm.algorithms
import hmm.concurrent
import hmm.utility
import time as t

# initial conditions
A, B, pi = hmm.utility.get_models()['t2']

print 'Read data ...'

obs = [
     numpy.loadtxt('data/hmm1.1000000.dat', dtype=numpy.int16),
#    numpy.loadtxt('data/hmm1.100.dat'),
#    numpy.array([1, 0, 1, 0, 1], dtype=numpy.int16)
]

obs2 = [
	numpy.loadtxt('data/t1.100000.dat', dtype=numpy.int16),
] * 10
 
maxit = 10
accuracy = 1e-3

numpy.set_printoptions(suppress=True)


kernels = [ hmm.kernel.fortran, hmm.kernel.c ] # hmm.kernel.python, hmm.kernel.c ]

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
        A1, B1, pi1, prob, it = \
            hmm.algorithms.baum_welch(
                ob, A, B, pi, 
                maxit=maxit, kernel=kernel, accuracy=accuracy, dtype=numpy.float32
            )
        print A1
        print B1
        print pi1
        end = t.time()
        time_used += end-start
        print 'log P( O | A,B,pi ): ', prob
        print 'iterations done: ', maxit
    print 'time used: ', time_used, ' seconds.'
    print

print 'perform Baum-Welch multiple algorithm'
print 'observation lengths: ', [ len(ob) for ob in obs2 ]
print 'tested kernel: ', kernels
print 'maximal iterations: ', maxit
print 'accuracy: ', accuracy
for kernel in kernels:
    print 'kernel: ', kernel
    start = t.time()
    A1, B1, pi1, prob, it = \
        hmm.algorithms.baum_welch_multiple(
            obs2, A, B, pi, 
            maxit=maxit, kernel=kernel, accuracy=accuracy, dtype=numpy.float32
        )
    print A1
    print B1
    print pi1
    end = t.time()
    print 'log P( O | A,B,pi ): ', prob
    print 'iterations done: ', maxit
    print 'time used: ', end-start, ' seconds.'
    print

print 'perform parallel version of Baum-Welch multiple algorithm'
print 'observation lengths: ', [ len(ob) for ob in obs2 ]
print 'tested kernel: ', kernels
print 'maximal iterations: ', maxit
print 'accuracy: ', accuracy
for kernel in kernels:
    print 'kernel: ', kernel
    start = t.time()
    A1, B1, pi1, prob, it = \
        hmm.concurrent.baum_welch_multiple(
            obs2, A, B, pi, 
            maxit=maxit, kernel=kernel, accuracy=accuracy
        )
    print A1
    print B1
    print pi1
    end = t.time()
    print 'log P( O | A,B,pi ): ', prob
    print 'iterations done: ', maxit
    print 'time used: ', end-start, ' seconds.'
    print
