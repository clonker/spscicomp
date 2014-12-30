import numpy as np
from PySimpleHMM import *


class NumPyHMM(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(NumPyHMM, self).__init__(N, M, A, B, pi)
		
	def forward(self, obs):
		T, N = len(obs), self.N
		alpha = np.zeros((T,N), dtype=np.double)
		self.scale = np.zeros(T, dtype=np.double)
		
		alpha[0] = self.pi * self.B[:, obs[0]]
		self.scale[0] = alpha[0].sum()
		alpha[0] /= self.scale[0]
		
		for t in range(T-1):
			alpha[t+1] = self.A.T.dot(alpha[t]) * self.B[:, obs[t+1]]
			self.scale[t+1] = alpha[t+1].sum()
			alpha[t+1] /= self.scale[t+1]
			
		return (np.sum(np.log(self.scale)), alpha)
		
	def backward(self, obs):
		T, N = len(obs), self.N
		beta = np.zeros((T,N), dtype=np.double)
		
		beta[T-1] = 1.0 / self.scale[T-1]
		
		for t in range(T-2, -1, -1):
			beta[t] = self.A.dot(self.B[:,obs[t+1]]*beta[t+1])
			beta[t] /= self.scale[t]
		
		return beta
		
	def computeGamma(self, alpha, beta):
		gamma = alpha * beta
		gamma = (gamma.T / gamma.sum(axis=1)).T
		return gamma
		
	def computeXi(self, obs, alpha, beta):
		T, N = len(obs), self.N
		xi = np.zeros((T-1,N,N), dtype=np.double)
		
		for t in range(T-1):
			sum = 0.0
			for i in range(N):
				for j in range(N):
					xi[t,i,j] = alpha[t,i]*self.A[i,j]*self.B[j,obs[t+1]]*beta[t+1,j]
					sum += xi[t,i,j]
			for i in range(N):
				for j in range(N):
					xi[t,i,j] /= sum
#			xi[t] = (self.A * alpha[t]).T * beta[t+1] * self.B[:,obs[t+1]]
#			xi[t] /= xi[t].sum()
			
		return xi
		
	def update(self, obs, gamma, xi):
		self.pi = gamma[0]
		xi_sum = xi.sum(axis=0)
		self.A = (xi_sum.T / xi_sum.sum(axis=1)).T
		for k in range(0, len(self.B[0])):
			self.B[:,k] = np.sum( ((obs == k) * gamma.T), axis=1 )
		self.B = (self.B.T / np.sum(self.B, axis=1)).T

