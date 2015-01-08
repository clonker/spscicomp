import unittest
import pyopencl
import numpy
import string
import hmm.kernel.python

ob = numpy.array([1, 0, 1, 1, 0, 1], dtype=numpy.int16)

A = numpy.array(
    [[ 0.333, 0.333 ],
     [ 0.333, 0.333 ]],
     dtype=numpy.float32)

B = numpy.array(
    [[ 0.7, 0.3 ],
     [ 0.5, 0.5 ]], dtype=numpy.float32)

pi = numpy.array([ 0.333, 0.333 ], dtype=numpy.float32)

T = len(ob)
N = len(A)
M = len(B[0])

FORWARD_SOURCE  = 'lib/opencl/forward_blelloch.cl'

numpy.set_printoptions(precision=3)

platform = pyopencl.get_platforms()[0]
device   = platform.get_devices(device_type=pyopencl.device_type.GPU)[0]
context  = pyopencl.Context([device])
f        = open(FORWARD_SOURCE, 'r')
source   = string.Template("".join(f.readlines())).substitute(N=N, M=M, precision="float")
kernel   = pyopencl.Program(context, source).build()
queue    = pyopencl.CommandQueue(context)
mf       = pyopencl.mem_flags

A_  = pyopencl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_  = pyopencl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
pi_ = pyopencl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
ob_ = pyopencl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)

def calc_matrices(A, B, ob, T):
	C = numpy.zeros((T,N,N), dtype=numpy.float32)
	C[0] = numpy.eye(N)
	for t in xrange(1,int(T)):
		for i in xrange(N):
			for j in xrange(N):
				C[t,i,j] = B[i, ob[t]] * A[j,i]
	return C

def downsweep(C, T):
	depth = int(T)/2
	offset = 1
	A = C.copy()
	while depth > 0:
		for worker in range(depth):
			left = offset * (2*worker+2) - 1
			right = offset * (2*worker+1) - 1
			A[left] = A[left].dot(A[right])
		depth /= 2
		offset *= 2
	return A

def upsweep(C, T):
	depth = 1
	offset = T/2
	A = C.copy()
	while depth < T:
		for worker in range(depth):
			left = offset * (2*worker+2) - 1
			right = offset * (2*worker+1) - 1
			tmp = A[left].copy()
			A[left] = A[right].dot(tmp)
			A[right] = tmp
		depth *= 2
		offset /= 2
	return A

def padding_time(T, NUM_GROUP_UNITS):
	return ((T - 1) / (2*NUM_GROUP_UNITS) + 1) * 2 * NUM_GROUP_UNITS

def forward(NUM_GROUP_UNITS, NUM_GROUPS):

	padded = padding_time(T, NUM_GROUP_UNITS)

	alpha_     = numpy.zeros((T,N),   dtype=numpy.float32)
	matrices_  = numpy.zeros((padded,N,N), dtype=numpy.float32)

	alpha    = pyopencl.Buffer(context, mf.READ_WRITE, alpha_.nbytes)
	matrices = pyopencl.Buffer(context, mf.READ_WRITE, matrices_.nbytes)
	scratch  = pyopencl.LocalMemory(4*N*N * 2*NUM_GROUP_UNITS )

	kernel.forward_build_matrices(queue, (NUM_GROUP_UNITS*NUM_GROUPS,), (NUM_GROUP_UNITS,),
		matrices, alpha, A_, B_, pi_, ob_, numpy.uint64(T))

	if padded - T > 0:
		kernel.append_identity(queue, (NUM_GROUP_UNITS*NUM_GROUPS,), (NUM_GROUP_UNITS,),
			matrices, numpy.uint64(T), numpy.uint64(padded - T))

#	pyopencl.enqueue_copy(queue, alpha_, alpha)
#	print 'alpha before:\n', alpha_

	stack = [ ]
	current = matrices		
	S = padded
	while S > 1:
#		print S
		new_S = S / (2*NUM_GROUP_UNITS)
		if new_S > 1:
			new_padded = padding_time(new_S, NUM_GROUP_UNITS)
		else:
			new_padded = 1
		# print new_padded
		# current_ = numpy.zeros((S, N, N), dtype=numpy.float32)
		# pyopencl.enqueue_copy(queue, current_, current)
		# print current_
		reduced  = pyopencl.Buffer(context, mf.READ_WRITE, new_padded*4*N*N)
		kernel.forward_reduce(queue, (NUM_GROUP_UNITS*NUM_GROUPS,), (NUM_GROUP_UNITS,), 
			reduced, current, scratch, numpy.uint64(S))
		if new_padded - new_S > 0:
			kernel.append_identity(queue, (NUM_GROUP_UNITS*NUM_GROUPS,), (NUM_GROUP_UNITS,),
				reduced, numpy.uint64(new_S), numpy.uint64(new_padded - new_S))
	 	
		# current_ = numpy.zeros((S, N, N), dtype=numpy.float32)
		# pyopencl.enqueue_copy(queue, current_, current)
		# print current_

	 	stack.append( (current, S) )
	 	current = reduced
	 	S = new_padded
	
	# reverse here
	reduced, rS = stack.pop()
	# reduced_ = numpy.zeros((rS, N, N), dtype=numpy.float32)
	# pyopencl.enqueue_copy(queue, reduced_, reduced)
	# print 'reduced:\n', reduced_
	while stack:
		# reduced_ = numpy.zeros((rS, N, N), dtype=numpy.float32)
		# pyopencl.enqueue_copy(queue, reduced_, reduced)
		# print 'reduced:\n', reduced_
		extended, S = stack.pop()
#		extended_ = numpy.zeros((S, N, N), dtype=numpy.float32)
#		pyopencl.enqueue_copy(queue, extended_, extended)
#	  	print 'extended before:\n', extended_
	  	kernel.forward_collect(queue, (NUM_GROUP_UNITS*NUM_GROUPS,), (NUM_GROUP_UNITS,),
	  		extended, reduced, numpy.uint64(S))
#	  	pyopencl.enqueue_copy(queue, extended_, extended)
#	  	print 'extended after:\n',extended_
	  	reduced = extended
	  	rS = S 

	kernel.forward_build_alpha(queue, (NUM_GROUP_UNITS*NUM_GROUPS,), (NUM_GROUP_UNITS,),
		alpha, reduced, numpy.uint64(T))
	pyopencl.enqueue_copy(queue, alpha_, alpha)
#	print 'alpha after:\n', alpha_
	_, alpha2 = hmm.kernel.python.forward_no_scaling(A, B, pi, ob)
#	print 'alpha should be:\n', alpha2
	numpy.testing.assert_almost_equal(alpha_, alpha2)

class TestForwardDemo(unittest.TestCase):

	def test_forward_build_matrices(self):
		NUM_GROUPS      = 1
		NUM_GROUP_UNITS = T

		alpha     = numpy.zeros((T,N),   dtype=numpy.float32)
		matrices  = numpy.zeros((T,N,N), dtype=numpy.float32)
		alpha_    = pyopencl.Buffer(context, mf.READ_WRITE, alpha.nbytes)
		matrices_ = pyopencl.Buffer(context, mf.READ_WRITE, matrices.nbytes)

		kernel.forward_build_matrices(queue, (NUM_GROUP_UNITS*NUM_GROUPS,), None,
			matrices_, alpha_, A_, B_, pi_, ob_, numpy.uint64(T))

		pyopencl.enqueue_copy(queue, matrices, matrices_)
		C = calc_matrices(A, B, ob, T)
		numpy.testing.assert_almost_equal(C, matrices)

	def test_forward_some_configurations(self):
		forward(1,2)
		forward(4,1)
		forward(1,1)
		forward(1,8)
		forward(4,3)