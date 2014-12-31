import numpy as np
from PySimpleHMM import *
from extension import hmm_fortran as ext

class FortranHMM(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(FortranHMM, self).__init__(N, M, A, B, pi)

	def forward(self, obs):
		alpha, self.scale, logprob  = ext.forward(self.A, self.B, self.pi, obs)
		return (logprob, alpha)
	
	def backward(self, obs):
		return ext.backward(self.A, self.B, obs, self.scale)

	def computeGamma(self, alpha, beta):
		return ext.computegamma(alpha, beta)
		
	def computeXi(self, obs, alpha, beta):
		return ext.computexi(self.A, self.B, obs, alpha, beta)
		
	def computeNominatorA(self, obs, alpha, beta):
		return ext.computenoma(self.A, self.B, obs, alpha, beta)
		
	def computeDenominatorA(self, gamma):
		return ext.computedenoma(gamma)
		
	def computeNominatorB(self, obs, gamma):
		return ext.computenomb(obs, gamma, len(self.B[0]))
		
	def update(self, obs, gamma, xi):
		self.A, self.B, self.pi = ext.update(obs, gamma, xi, len(self.B[0]))
