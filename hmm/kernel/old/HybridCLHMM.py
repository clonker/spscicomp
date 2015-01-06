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

#    @profile
    def BaumWelch(self, ob, accuracy, maxit):
        old_eps = 0.0
        it = 0
        T,N = len(ob),self.N
        mf = cl.mem_flags
        
        new_eps,alpha = self.forward(ob)
        
        gamma = np.zeros_like(alpha)
        xi = np.zeros((T,N,N), dtype=self.dtype)
        
        ob_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
        gamma_buf = cl.Buffer(self.context, mf.READ_WRITE, gamma.nbytes)
        xi_buf = cl.Buffer(self.context, mf.READ_WRITE, xi.nbytes)
        loc_buf = cl.LocalMemory(800*N);
        
        while (abs(new_eps - old_eps) >= accuracy and it < maxit):
            beta = self.backward(ob)
            A_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.A)
            B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.B)
            alpha_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=alpha)
            beta_buf  = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=beta)
            self.program.compute_gamma(self.queue, (T,), alpha_buf, beta_buf, gamma_buf)
            self.program.compute_xi(self.queue, (T-1,), A_buf, B_buf, ob_buf, alpha_buf, beta_buf, xi_buf);
            self.program.new_A(self.queue, (200*N,200,), xi_buf, A_buf, loc_buf, np.int32(T))
            self.program.new_B(self.queue, (200*N,200,), gamma_buf, ob_buf, B_buf, loc_buf, np.int32(T))

            cl.enqueue_copy(self.queue, self.A, A_buf)
            cl.enqueue_copy(self.queue, self.B, B_buf)

            self.pi = alpha[0]*beta[0]; self.pi /= self.pi.sum();
            old_eps = new_eps
            new_eps, alpha = self.forward(ob)
            it += 1
        return (new_eps, it)    
