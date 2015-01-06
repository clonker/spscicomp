import numpy as np
from CHMM32 import *
import pyopencl as cl
import multiprocessing
from string import Template

class PyOpenCLHMM(CHMM32):
    def __init__(self, N, M, A, B, pi):
        super(PyOpenCLHMM, self).__init__(N, M, A, B, pi)
        self.platform = cl.get_platforms()[0]        
        # fetch GPU
        devices = self.platform.get_devices(device_type=cl.device_type.GPU)
        if (len(devices) == 0):
            raise ValueError
        self.device = devices[0]
        # create context for this object (own GPU world)
        self.context = cl.Context([self.device])        
        # now compile the program
        f = open('../extension/multiple.cl', 'r')
        fstr = Template("".join(f.readlines())).substitute(N=self.N, M=self.M)
        self.kernel = cl.Program(self.context, fstr).build()
        self.queue = cl.CommandQueue(self.context)
        self.TOTAL_COMPUTE_UNITS = 4096
        self.TOTAL_WORKGROUPS = 32
        self.LOCAL_COMPUTE_UNITS = self.TOTAL_COMPUTE_UNITS / self.TOTAL_WORKGROUPS

    def BaumWelch_multiple(self, obs, accuracy, maxiter):
        K, N, M = np.int32(len(obs)), self.N, self.M
        Ts = np.cumsum([ len(ob) for ob in obs ])
        Tss = np.zeros(K+1, np.int64)
        Tss[1:] = Ts
        T  = int(Ts[K-1])


        mf = cl.mem_flags

        obs_array = np.zeros(T, dtype=np.int32)
        for k in xrange(K):
            obs_array[Tss[k]:Tss[k+1]] = obs[k]

        # constant buffers to be read from
        os = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=obs_array)
        ts = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Tss)        
        pi = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pi)
        
        
        # results are placed here
        reduced_xi = np.zeros((self.TOTAL_WORKGROUPS,N,N), dtype=self.dtype)
        reduced_bc = np.zeros((self.TOTAL_WORKGROUPS,N,M), dtype=self.dtype)

        itemsize = self.A.itemsize

        # buffers to be written to
        C       = cl.Buffer(self.context, mf.READ_WRITE, T*N*N*itemsize)
        alpha   = cl.Buffer(self.context, mf.READ_WRITE, T*N*itemsize)
        beta    = cl.Buffer(self.context, mf.READ_WRITE, T*N*itemsize)
        weights = cl.Buffer(self.context, mf.READ_WRITE, N*itemsize)
        gamma   = cl.Buffer(self.context, mf.READ_WRITE, T*N*itemsize)
        xi      = cl.Buffer(self.context, mf.READ_WRITE, (T-1)*self.A.nbytes)
        bc      = cl.Buffer(self.context, mf.READ_WRITE, T*self.B.nbytes)
        rb      = cl.Buffer(self.context, mf.WRITE_ONLY, reduced_bc.nbytes)
        rxi     = cl.Buffer(self.context, mf.WRITE_ONLY, reduced_xi.nbytes)        
        scratch = cl.LocalMemory(self.LOCAL_COMPUTE_UNITS * N * N * itemsize)      
        
        old_eps = 0.0
        it = 0
        new_eps = accuracy+1
        while (abs(new_eps - old_eps) > accuracy and it < maxiter):

            A = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.A)
            B = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.B)

            self.kernel.prepare_forward(self.queue, (T,), A, B, pi, os, C)
            self.kernel.reduce_forward(self.queue, (T,), C, alpha, weights)
            self.kernel.prepare_backward(self.queue, (T,), A, B, os, C, ts, K)
            self.kernel.reduce_backward(self.queue, (T,), C, beta, weights, ts, K)
            self.kernel.make_gamma(self.queue, (T,), alpha, beta, gamma)
            self.kernel.make_xi(self.queue, (T,), A, B, alpha, beta, xi, weights, os, ts, K)
            self.kernel.make_bc(self.queue, (T,), gamma, bc, os)

            self.kernel.reduce_xi(
                    self.queue,
                    (self.TOTAL_COMPUTE_UNITS,),
                    (self.LOCAL_COMPUTE_UNITS,),
                    xi, rxi, scratch, ts, K
            )
            self.kernel.reduce_bc(
                    self.queue,
                    (self.TOTAL_COMPUTE_UNITS,),
                    (self.LOCAL_COMPUTE_UNITS,),
                    bc, rb, scratch, np.int64(T)
            )

            cl.enqueue_copy(self.queue, reduced_xi, rxi)
            cl.enqueue_copy(self.queue, reduced_bc, rb)
   
            self.A = reduced_xi.sum(axis=0)
            self.A = (self.A.T / self.A.sum(axis=1)).T
            self.B = reduced_bc.sum(axis=0)
            self.B = (self.B.T / self.B.sum(axis=1)).T

            if (it == 0):
                old_eps = 0
            else:
                old_eps = new_eps
            new_eps = 0
            it += 1
            
            
        return (new_eps, it) 
