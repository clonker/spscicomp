#!/usr/bin/python

import numpy as np
import hmm.kernel.opencl as hcl
import hmm.kernel.c as hc
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

ob = np.loadtxt('data/t1.3000000.dat')

N, M, T = len(A), len(B[0]), len(ob)
 
maxiter = 100

print 'Initialize OpenCL'
ctx = hcl.Context(N, M)

print 'start opencl implementation of forward algorithm...'
start = t.time()
for it in range(maxiter):
	hcl.forward_no_scaling_naive(ctx, A, B, pi, ob)
end = t.time()
print '{d} iterations took us {t} seconds.'.format(d=maxiter, t=(end-start))

print 'start opencl implementation of forward algorithm...'
start = t.time()
for it in range(maxiter):
	hcl.forward_no_scaling_belloch(ctx, A, B, pi, ob)
end = t.time()
print '{d} iterations took us {t} seconds.'.format(d=maxiter, t=(end-start))

print 'start c implementation of forward algorithm...'
start = t.time()
for it in range(maxiter):
	hc.forward(A, B, pi, ob)
end = t.time()
print '{d} iterations took us {t} seconds.'.format(d=maxiter, t=(end-start))