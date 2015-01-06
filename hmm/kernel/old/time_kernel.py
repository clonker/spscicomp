#!/usr/bin/python

import hmm.kernel
import hmm.utilities
import pyopencl as cl
import numpy as np
import time as t

# initial conditions
A = np.array(
    [[ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ],
     [ 0.333, 0.333, 0.333 ]])   
B = np.array(
	[[ 1.0, 0.0 ],
     [ 0.5, 0.5 ],
     [ 0.0, 1.0 ]])    
Pi = np.array(
	[ 0.333, 0.333, 0.333])

ob_lens    = [ 10000 ]
iterations = 10000
N = len(Pi)

LOCAL_SIZE  = 256
WORK_GROUPS = 14

hmm_CL = hmm.kernel.OpenCL(A, B, Pi)
hmm_C  = hmm.kernel.C32(A,B,Pi)

mf    = cl.mem_flags
a     = cl.Buffer(hmm_CL.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
b     = cl.Buffer(hmm_CL.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
pi    = cl.Buffer(hmm_CL.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Pi)
scratch = cl.LocalMemory(LOCAL_SIZE*N*N*4)

for T in ob_lens:
	Ob = np.loadtxt('tests/data/t1.{d}.dat'.format(d=T), dtype=np.int16)

	ob    = cl.Buffer(hmm_CL.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Ob)
	alpha = cl.Buffer(hmm_CL.context, mf.WRITE_ONLY, T*N*4)
	c     = cl.Buffer(hmm_CL.context, mf.READ_WRITE, T*N*N*4)
	Alpha = np.zeros((T,N), dtype=np.float32)
	start = t.time()
	for it in range(iterations):        
		hmm_CL.kernel.forward(hmm_CL.queue,(WORK_GROUPS*LOCAL_SIZE,),(LOCAL_SIZE,),alpha, c, a, b, pi, ob, scratch, np.uint64(T))
        cl.enqueue_copy(hmm_CL.queue, Alpha, alpha)        
	end = t.time()
	print 'Used time: {time}'.format(time=end-start)
	start = t.time()
	for it in range(iterations):        
		hmm_C.forward(np.asarray(Ob, dtype=np.int32))
	end = t.time()
	print 'Used time: {time}'.format(time=end-start)