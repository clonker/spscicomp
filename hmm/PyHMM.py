import numpy as np
from PySimpleHMM import *


class PyHMM(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(PyHMM, self).__init__(N, M, A, B, pi)

	def forward(self, obs):
		T, N = len(obs), self.N
		alpha = np.zeros((T,N), dtype=np.float64)
		self.scale = np.zeros(T, dtype=np.float64)
		for i in range(N):
			alpha[0,i] = self.pi[i]*self.B[i,obs[0]]
			self.scale[0] += alpha[0,i]
		for i in range(N):
			alpha[0,i] /= self.scale[0]
		for t in range(T-1):
			for j in range(N):
				alpha[t+1,j] = 0.0
				for i in range(N):
					alpha[t+1,j] += alpha[t,i] * self.A[i,j]
				alpha[t+1,j] *= self.B[j,obs[t+1]]
				self.scale[t+1] += alpha[t+1,j]
			for j in range(N):
				alpha[t+1,j] /= self.scale[t+1]
		logprob = 0.0
		for t in range(T):
			logprob += np.log(self.scale[t])
		return (logprob, alpha)

	def backward(self, obs):
		T, N = len(obs), self.N
		beta = np.zeros((T,N), dtype=np.float64)
		for i in range(N):
			beta[T-1,i] = 1.0 / self.scale[T-1]
		for t in range(T-2, -1, -1):
			for i in range(N):
				beta[t,i] = 0.0
				for j in range(N):
					beta[t,i] += self.A[i,j] * beta[t+1,j] * self.B[j,obs[t+1]] / self.scale[t]
		return beta
