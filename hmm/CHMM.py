import numpy as np
from PySimpleHMM import *
from extension import hmm_ext as ext

class CHMM(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(CHMM, self).__init__(N, M, A, B, pi)
		
	def forward(self, obs):
		alpha = np.zeros((len(obs),self.N), dtype=np.double)
		self.scale = np.zeros(len(obs), dtype=np.double)
		logprob = ext.forward(self.A, self.B, self.pi, obs, alpha, self.scale)
		return (logprob, alpha)
	
	def backward(self, obs):
		beta = np.zeros((len(obs),self.N), dtype=np.double)
		ext.backward(self.A, self.B, self.pi, obs, beta, self.scale)
		return beta
		
	def computeGamma(self, alpha, beta):
		gamma = np.zeros(alpha.shape, dtype=np.double)
		ext.compute_gamma(alpha, beta, gamma)
		return gamma
		
	def copmuteXi(self, obs, alpha, beta):
		xi = np.zeros((len(obs),self.N,self.N), dtype=np.double)
		ext.compute_xi(self.A, self.B, self.pi, obs, alpha, beta, xi)
		return xi
		
	def update(self, obs, gamma, xi):
		ext.update(self.A, self.B, self.pi, obs, gamma, xi)
		
	def BaumWelch(self, obs, accuracy, maxit):
		T,N = len(obs), self.N
		old_prob = 0.0
		it = 0
		alpha = np.zeros((T,N),dtype=np.double)
		scale = np.zeros((T),dtype=np.double)
		beta  = np.zeros((T,N),dtype=np.double)
		gamma = np.zeros((T,N),dtype=np.double)
		xi    = np.zeros((T-1,N,N),dtype=np.double)
		
		new_prob = ext.forward(self.A, self.B, self.pi, obs, alpha, scale)
		while (abs(new_prob - old_prob) > accuracy and it < maxit):
			ext.backward(self.A, self.B, self.pi, obs, beta, scale)
			ext.compute_gamma(alpha, beta, gamma)
			ext.compute_xi(self.A, self.B, self.pi, obs, alpha, beta, xi)
			ext.update(self.A, self.B, self.pi, obs, gamma, xi)
			self.update(obs, gamma, xi)
			old_prob = new_prob
			new_prob = ext.forward(self.A, self.B, self.pi, obs, alpha, scale)
			it += 1
		return (new_prob, it)
