#!/usr/bin/python

import numpy as np
import hmm.kernel.opencl
import hmm.kernel.c 
import time as t

# initial conditions
A = np.array(
    [[ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ]], dtype=np.float32)

B = np.array(
    [[ 1.0, 0.0 ],
     [ 0.5, 0.5 ],
     [ 0.0, 1.0 ]], dtype=np.float32)

pi = np.array([ 0.333, 0.333, 0.333 ], dtype=np.float32)

print 'Read data ...'

ob = np.loadtxt('data/t1.100000.dat')

N, M, T = len(A), len(B[0]), len(ob)
 
maxiter = 1000

print 'Initialize OpenCL'
ctx = hmm.kernel.opencl.Context(N, M)

print 'start opencl implementation of forward algorithm...'
start = t.time()
for it in range(maxiter):
	hmm.kernel.opencl.forward_no_scaling_naive(ctx, A, B, pi, ob)
end = t.time()
print '{d} iterations took us {t} seconds.'.format(d=maxiter, t=(end-start))

print 'start c implementation of forward algorithm...'
start = t.time()
for it in range(maxiter):
	hmm.kernel.c.forward(A, B, pi, ob)
end = t.time()
print '{d} iterations took us {t} seconds.'.format(d=maxiter, t=(end-start))