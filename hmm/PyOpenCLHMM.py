import numpy as np
from CHMM32 import *
import pyopencl as cl

class PyOpenCLHMM(CHMM32):
	def __init__(self, N, M, A, B, pi):
		super(PyOpenCLHMM, self).__init__(N, M, A, B, pi)
		
	#@profile
	def BaumWelch_multiple(self, obs, accuracy, maxiter):

		# initialize OpenCL
		platform = cl.get_platforms()[0]
		device = platform.get_devices()[1]
		context = cl.Context([device])
		f = open('../extension/hmm.cl', 'r')
		fstr = "".join(f.readlines())
		program = cl.Program(context, fstr).build()
		queue = cl.CommandQueue(context)
		
		
		# create buffer
		K, N, M = len(obs), self.N, self.M
		T = np.asarray([len(ob) for ob in obs], dtype=np.int32)
		mf = cl.mem_flags
		os = np.asarray(obs, dtype=np.int32)
		nomsA = np.zeros((K,N,N), dtype=np.float32)
		denomsA = np.zeros((K,N), dtype=np.float32)
		nomsB = np.zeros((K,N,M), dtype=np.float32)
		denomsB = np.zeros((K,N), dtype=np.float32)
		maxT = max(T)
		obs_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=os)
		T_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T)
		# local buffers
		alpha = np.zeros((K,maxT,N), dtype=np.float32)
		scale = np.zeros((K,maxT), dtype=np.float32)
		beta = np.zeros((K,maxT,N), dtype=np.float32)
		gamma = np.zeros((K,maxT,N), dtype=np.float32)		
		nomA = np.zeros((K,N,N), dtype=np.float32)
		denomA = np.zeros((K,N), dtype=np.float32)
		weights = np.zeros(K, dtype=np.float32)		
		nomsA_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=nomsA)
		denomsA_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=denomsA)
		nomsB_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=nomsB)
		denomsB_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=denomsB)
		weights_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)		
		alpha_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=alpha)
		scale_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=scale)
		beta_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=beta)
		gamma_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gamma)
		nomA_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nomA)
		denomA_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=denomA)
		
		# start iteration loop
		
		old_eps = 0.0
		it = 0
		new_eps = accuracy+1

		while (abs(new_eps - old_eps) > accuracy and it < maxiter):
			A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.A)
			B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.B)
			pi_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pi)
			program.process_obs(queue, (K,1,), None, weights_buf, \
				nomsA_buf, denomsA_buf, nomsB_buf, denomsB_buf, A_buf, B_buf, pi_buf, \
				obs_buf, np.int32(N), np.int32(M), np.int32(maxT), T_buf, \
				alpha_buf, scale_buf, beta_buf, gamma_buf, nomA_buf)
			cl.enqueue_copy(queue, weights, weights_buf)
			cl.enqueue_copy(queue, nomsB, nomsB_buf)
			cl.enqueue_copy(queue, nomsA, nomsA_buf)
			cl.enqueue_copy(queue, denomsA, denomsA_buf)
			cl.enqueue_copy(queue, denomsB, denomsB_buf)
			self.update_multiple(weights, nomsA, denomsA, nomsB, denomsB)

			if (it == 0):
				old_eps = 0
			else:
				old_eps = new_eps
			new_eps = np.sum(weights)
			it += 1

		return new_eps, it
		
