import numpy as np
from CHMM32 import *
import pyopencl as cl
from string import Template

class HybridCLHMM(CHMM32):
	def __init__(self, N, M, A, B, pi):
		# cast to float because My graphics card can not handle doubles
		super(HybridCLHMM, self).__init__(N, M, A, B, pi)
		# TODO select correct platform if multiple are given
		self.platform = cl.get_platforms()[0]		
		# fetch GPU
		devices = self.platform.get_devices()#device_type=cl.device_type.GPU)
		if (len(devices) == 0):
			# TODO look for better exception
			raise ValueError
		# TODO select best GPU...
		self.device = devices[0]
		# create context for this object (own GPU world)
		self.context = cl.Context([self.device])		
		# now compile the program
		f = open('../extension/hybrid_hmm.cl', 'r')
		fstr = Template("".join(f.readlines())).substitute(N=self.N, M=self.M)
		self.program = cl.Program(self.context, fstr).build()
		self.queue = cl.CommandQueue(self.context)

	@profile	
	def computeGamma(self, alpha, beta):
		T, N = len(alpha), self.N
		mf = cl.mem_flags
		gamma = np.zeros((T,N), dtype=np.float32)
		alpha_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=alpha)
		beta_buf  = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=beta)
		gamma_buf = cl.Buffer(self.context, mf.WRITE_ONLY, gamma.nbytes)
		gamma_t   = cl.LocalMemory(int(N*np.dtype('float32').itemsize))
		self.program.compute_gamma(self.queue, (T*N,), (N,), alpha_buf, beta_buf, gamma_buf, gamma_t)
		cl.enqueue_copy(self.queue, gamma, gamma_buf)

		return gamma

