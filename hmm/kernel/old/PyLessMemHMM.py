import numpy as np
from PySimpleHMM import *


class PyLessMemHMM(PySimpleHMM):
    def __init__(self, N, M, A, B, pi):
        super(PyLessMemHMM, self).__init__(N, M, A, B, pi)
        
    def computeXi(self, obs, alpha, beta):
        T, N = len(obs), self.N
        xi = np.zeros((N,N), dtype=np.float64)
        xi_t = np.zeros((N,N), dtype=np.float64)
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
        
    def update(self, obs, gamma, xi):
        T,N,M = len(obs),self.N,self.M
        for i in range(N):
            self.pi[i] = gamma[0,i]
        for i in range(N):
            gamma_sum = 0.0
            for t in range(T-1):
                gamma_sum += gamma[t,i]
            for j in range(N):
                self.A[i,j] = xi[i,j] / gamma_sum
            gamma_sum += gamma[T-1, i]
            for k in range(M):
                self.B[i,k] = 0.0
                for t in range(T):
                    if obs[t] == k:
                        self.B[i,k] += gamma[t,i]
                self.B[i,k] /= gamma_sum
