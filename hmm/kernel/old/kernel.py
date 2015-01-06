""" Python Hidden Markov Model kernels

Algorithms as proposed by L. Rabiner (see http://dx.doi.org/10.1109/5.18626 )

"""
import numpy as np
import utilities
import pyopencl as cl
import string
from extension import hmm_ext as cext

class Python(object):
    def __init__(self, A, B, pi, dtype=np.float64, scaling=True):
        self.N = len(A)
        self.M = len(B[0])
        self.dtype = dtype
        self.A = np.asarray(A.copy(), dtype=self.dtype)
        self.B = np.asarray(B.copy(), dtype=self.dtype)
        self.pi = np.asarray(pi.copy(), dtype=self.dtype)

    def forward(self, obs):
        T, N = len(obs), self.N
        alpha = np.zeros((T,N), dtype=self.dtype)
        for i in range(N):
            alpha[0,i] = self.pi[i]*self.B[i,obs[0]]
        for t in range(T-1):
            for j in range(N):
                alpha[t+1,j] = 0.0
                for i in range(N):
                    alpha[t+1,j] += alpha[t,i] * self.A[i,j]
                alpha[t+1,j] *= self.B[j,obs[t+1]]
        prob = 0.0
        for i in range(N):
            prob += alpha[T-1,i]
        return (np.log(prob), alpha)

    def backward(self, obs):
        T, N = len(obs), self.N
        beta = np.zeros((T,N), dtype=self.dtype)
        for i in range(N):
            beta[T-1,i] = 1
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t,i] = 0.0
                for j in range(N):
                    beta[t,i] += self.A[i,j] * beta[t+1,j] * self.B[j,obs[t+1]]
        return beta

    def gamma(self, alpha, beta):
        T, N = len(alpha), self.N
        gamma = np.zeros((T,N), dtype=self.dtype)
        for t in range(T):
            sum = 0.0
            for i in range(N):
                gamma[t,i] = alpha[t,i]*beta[t,i]
                sum += gamma[t,i]
            for i in range(N):
                gamma[t,i] /= sum
        return gamma

    def xi(self, obs, alpha, beta):
        T, N = len(obs), self.N
        xi = np.zeros((T-1,N,N), dtype=self.dtype)
        for t in range(T-1):
            sum = 0.0
            for i in range(N):
                for j in range(N):
                    xi[t,i,j] = alpha[t,i]*self.A[i,j]*self.B[j,obs[t+1]]*beta[t+1,j]
                    sum += xi[t,i,j]
            for i in range(N):
                for j in range(N):
                    xi[t,i,j] /= sum
        return xi

    def nominator_A(self, obs, alpha, beta):
        T, N = len(obs), self.N
        xi = np.zeros((N,N), dtype=self.dtype)
        xi_t = np.zeros((N,N), dtype=self.dtype)
        for t in range(T-1):
            sum = 0.0
            for i in range(N):
                for j in range(N):
                    xi_t[i,j] = alpha[t,i]*self.A[i,j]*self.B[j,obs[t+1]]*beta[t+1,j]
                    sum += xi_t[i,j]
            for i in range(N):
                for j in range(N):
                    xi[i,j] += xi_t[i,j] / sum
        return xi

    def denominator_A(self, gamma):
        denom = np.zeros((self.N), dtype=self.dtype)
        for t in range(len(gamma)-1):
            for i in range(self.N):
                denom[i] += gamma[t,i]
        return denom

    def nominator_B(self, obs, gamma):
        B = np.zeros_like(self.B)
        for i in range(self.N):
            for k in range(self.M):
                for t in range(len(obs)):
                    if (obs[t] == k):
                        B[i,k] += gamma[t,i]
        return B

    def process_obs(self, obs):
        T = len(obs)
        weight, alpha = self.forward(obs)
        beta = self.backward(obs)
        gamma = self.gamma(alpha, beta)
        nomA = self.nominator_A(obs, alpha, beta)
        denomA = self.denominator_A(gamma)
        nomB = self.nominator_B(obs, gamma)
        denomB = denomA + gamma[T-1]
        return weight, nomA, denomA, nomB, denomB

    def update_multiple(self, weights, nomsA, denomsA, nomsB, denomsB):
        K, N, M = len(weights), self.N, self.M
        for i in range(N):
            nomA = np.zeros(N, dtype=self.dtype)
            denomA = 0.0
            for k in range(K):
                nomA += weights[k] * nomsA[k,i,:]
                denomA += weights[k] * denomsA[k,i]
            self.A[i,:] = nomA / denomA
        for i in range(N):
            nomB = np.zeros(M, dtype=self.dtype)
            denomB = 0.0
            for k in range(K):
                nomB += weights[k] * nomsB[k, i, :]
                denomB += weights[k] * denomsB[k, i]
            self.B[i,:] = nomB / denomB

    def baum_welch_multiple(self, obss, accuracy, maxiter):
        K, N, M = len(obss), self.N, self.M
        nomsA = np.zeros((K,N,N), dtype=self.dtype)
        denomsA = np.zeros((K,N), dtype=self.dtype)
        nomsB = np.zeros((K,N,M), dtype=self.dtype)
        denomsB = np.zeros((K,N), dtype=self.dtype)
        weights = np.zeros(K, dtype=self.dtype)        
        old_eps = 0.0
        it = 0
        new_eps = accuracy+1
        while (abs(new_eps - old_eps) > accuracy and it < maxiter):
            for k in range(K):
                weights[k], nomsA[k], denomsA[k], nomsB[k], denomsB[k] = self.process_obs(obss[k])
            self.update_multiple(weights, nomsA, denomsA, nomsB, denomsB)
            if (it == 0):
                old_eps = 0
            else:
                old_eps = new_eps
            new_eps = np.sum(weights)
            it += 1
        return new_eps, it

    def update(self, obs, gamma, xi):
        T,N,M = len(obs),self.N,self.M
        for i in range(N):
            self.pi[i] = gamma[0,i]
        for i in range(N):
            gamma_sum = 0.0
            for t in range(T-1):
                gamma_sum += gamma[t,i]
            for j in range(N):
                self.A[i,j] = 0.0
                for t in range(T-1):
                    self.A[i,j] += xi[t,i,j]
                self.A[i,j] /= gamma_sum
            gamma_sum += gamma[T-1, i]
            for k in range(M):
                self.B[i,k] = 0.0
                for t in range(T):
                    if obs[t] == k:
                        self.B[i,k] += gamma[t,i]
                self.B[i,k] /= gamma_sum

    def baum_welch(self, obs, accuracy, maxit):
        old_eps = 0.0
        it = 0
        T = len(obs)
        new_eps,alpha = self.forward(obs)
        while (abs(new_eps - old_eps) > accuracy and it < maxit):
            beta = self.backward(obs)
            gamma = self.gamma(alpha, beta)
            xi = self.xi(obs, alpha, beta)
            self.update(obs, gamma, xi)
            old_eps = new_eps
            new_eps, alpha = self.forward(obs)
            it += 1
        return (new_eps, it)
                        
    def print_model(self):
        """Print A, B and pi."""
        print 'A:\n', np.round(self.A, 2)
        print 'B:\n', np.round(self.B, 2)
        print 'pi:\n', np.round(self.pi, 2)

    def generate_seq(self, obsLength):
        """Creates a random Sequence of length n on base of this model."""
        obs = np.empty(obsLength, dtype='int')
        current = utilities.random_by_dist(self.pi)
        for i in range(obsLength):
            obs[i]  = utilities.random_by_dist(self.B[current,:])
            current = utilities.random_by_dist(self.A[current])
        return obs

class C(Python):
    def __init__(self, A, B, pi):
        super(C, self).__init__(A, B, pi)
        
    def forward(self, obs):
        logprob, alpha, self.scale  = cext.forward(self.A, self.B, self.pi, obs)
        return (logprob, alpha)
    
    def backward(self, ob):
        return cext.backward(self.A, self.B, ob, self.scale)

    def computeGamma(self, alpha, beta):
        return cext.compute_gamma(alpha, beta)

    def computeXi(self, obs, alpha, beta):
        return cext.compute_xi(self.A, self.B, obs, alpha, beta)
        
    def update(self, obs, gamma, xi):
        self.A, self.B, self.pi = cext.update(obs, gamma, xi, len(self.B[0]))

    def computeNominatorA(self, ob, alpha, beta):
        return cext.compute_nomA(self.A, self.B, ob, alpha, beta)
        
    def computeDenominatorA(self, gamma):
        return cext.compute_denomA(gamma)
        
    def computeNominatorB(self, ob, gamma):
        return cext.compute_nomB(ob, gamma, len(self.B[0]))

class C32(Python):
    def __init__(self, A, B, pi):
        super(C32, self).__init__(A, B, pi, np.float32)
        
    def forward(self, obs):
        logprob, alpha, self.scale  = cext.forward32(self.A, self.B, self.pi, obs)
        return (logprob, alpha)

    def backward(self, ob):
        return cext.backward32(self.A, self.B, ob, self.scale)

    def computeGamma(self, alpha, beta):
        return cext.compute_gamma32(alpha, beta)

    def computeXi(self, obs, alpha, beta):
        return cext.compute_xi32(self.A, self.B, obs, alpha, beta)
        
    def update(self, obs, gamma, xi):
        self.A, self.B, self.pi = ext.update32(obs, gamma, xi, len(self.B[0]))

    def computeNominatorA(self, ob, alpha, beta):
        return cext.compute_nomA32(self.A, self.B, ob, alpha, beta)
        
    def computeDenominatorA(self, gamma):
        return cext.compute_denomA32(gamma)
        
    def computeNominatorB(self, ob, gamma):
        return cext.compute_nomB32(ob, gamma, len(self.B[0]))

class OpenCL(Python):

    WORK_GROUPS = 2
    LOCAL_SIZE  = 3
    
    def __init__(self, A, B, pi):
        super(OpenCL, self).__init__(A, B, pi, dtype=np.float32)
        self.platform = cl.get_platforms()[0]        
        devices = self.platform.get_devices(device_type=cl.device_type.GPU)
        if (len(devices) == 0):
            raise ValueError
        self.device = devices[0]
        self.context = cl.Context([self.device])        
        f = open('./lib/opencl/forward.cl', 'r')
        fstr = string.Template("".join(f.readlines())).substitute(
                N=self.N, M=self.M, DEBUG=0)
        self.kernel = cl.Program(self.context, fstr).build()
        self.queue = cl.CommandQueue(self.context)

    def forward(self, ob):
        T, N = len(ob), self.N
        ob    = np.asarray(ob, dtype=np.int16)
        mf    = cl.mem_flags
        obs   = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)
        A     = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.A)
        B     = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.B)
        pi    = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pi)
        alpha = cl.Buffer(self.context, mf.WRITE_ONLY, T*N*4)
        C     = cl.Buffer(self.context, mf.READ_WRITE, T*N*N*4)
        debug = cl.Buffer(self.context, mf.WRITE_ONLY, self.WORK_GROUPS*self.LOCAL_SIZE*N*N*4)
        scratch = cl.LocalMemory(self.LOCAL_SIZE*N*N*4)

        self.kernel.forward(self.queue,
                (self.WORK_GROUPS*self.LOCAL_SIZE,),
                (self.LOCAL_SIZE,),
                alpha, C, A, B, pi, obs, scratch, np.uint64(T))

        al = np.zeros((T,N), self.dtype)
        cl.enqueue_copy(self.queue, al, alpha)
  
        return np.log(al[T-1].sum()), al
