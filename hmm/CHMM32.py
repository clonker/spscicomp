import numpy as np
from PySimpleHMM import *
from extension import hmm_ext as ext

class CHMM32(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(CHMM32, self).__init__(N, M, A, B, pi, np.float32)
		
	def forward(self, obs):
		logprob, alpha, self.scale  = ext.forward32(self.A, self.B, self.pi, obs)
		return (logprob, alpha)
	
	def backward(self, ob):
		return ext.backward32(self.A, self.B, ob, self.scale)

	@profile
	def computeGamma(self, alpha, beta):
		return ext.compute_gamma32(alpha, beta)
		
	def computeXi(self, obs, alpha, beta):
		return ext.compute_xi32(self.A, self.B, obs, alpha, beta)
		
	def update(self, obs, gamma, xi):
		self.A, self.B, self.pi = ext.update32(obs, gamma, xi, len(self.B[0]))

	def computeNominatorA(self, ob, alpha, beta):
		return ext.compute_nomA32(self.A, self.B, ob, alpha, beta)
		
	def computeDenominatorA(self, gamma):
		return ext.compute_denomA32(gamma)
		
	def computeNominatorB(self, ob, gamma):
		return ext.compute_nomB32(ob, gamma, len(self.B[0]))
		
#	def update_multiple(self, weights, nomsA, denomsA, nomsB, denomsB):
#		self.A, self.B = ext.update_multiple32(weights.copy(), nomsA.copy(), denomsA.copy(), nomsB.copy(), denomsB.copy())
