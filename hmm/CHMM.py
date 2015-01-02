import numpy as np
from PySimpleHMM import *
from extension import hmm_ext as ext

class CHMM(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(CHMM, self).__init__(N, M, A, B, pi, np.float64)
		
	def forward(self, obs):
		logprob, alpha, self.scale  = ext.forward(self.A, self.B, self.pi, obs)
		return (logprob, alpha)
	
	def backward(self, ob):
		return ext.backward(self.A, self.B, ob, self.scale)

	def computeGamma(self, alpha, beta):
		return ext.compute_gamma(alpha, beta)
		
	def computeXi(self, obs, alpha, beta):
		return ext.compute_xi(self.A, self.B, obs, alpha, beta)
		
	def update(self, obs, gamma, xi):
		self.A, self.B, self.pi = ext.update(obs, gamma, xi, len(self.B[0]))

	def computeNominatorA(self, ob, alpha, beta):
		return ext.compute_nomA(self.A, self.B, ob, alpha, beta)
		
	def computeDenominatorA(self, gamma):
		return ext.compute_denomA(gamma)
		
	def computeNominatorB(self, ob, gamma):
		return ext.compute_nomB(ob, gamma, len(self.B[0]))
		
#	def update_multiple(self, weights, nomsA, denomsA, nomsB, denomsB):
#		self.A, self.B = ext.update_multiple(weights.copy(), nomsA.copy(), denomsA.copy(), nomsB.copy(), denomsB.copy())
