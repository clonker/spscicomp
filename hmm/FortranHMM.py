import numpy as np
from PySimpleHMM import *
from extension import hmm_fortran as ext

class FortranHMM(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(FortranHMM, self).__init__(N, M, A, B, pi)

	def forward(self, ob):
		alpha, self.scale, logprob  = ext.forward(self.A, self.B, self.pi, ob)
		return (logprob, alpha)
	
	def backward(self, ob):
		return ext.backward(self.A, self.B, ob, self.scale)

	def computeGamma(self, alpha, beta):
		return ext.computegamma(alpha, beta)
		
	def computeXi(self, ob, alpha, beta):
		return ext.computexi(self.A, self.B, ob, alpha, beta)
		
	def computeNominatorA(self, ob, alpha, beta):
		return ext.computenoma(self.A, self.B, ob, alpha, beta)
		
	def computeDenominatorA(self, gamma):
		return ext.computedenoma(gamma)
		
	def computeNominatorB(self, ob, gamma):
		return ext.computenomb(ob, gamma, len(self.B[0]))
		
	def update(self, ob, gamma, xi):
		self.A, self.B, self.pi = ext.update(ob, gamma, xi, len(self.B[0]))
		
	def update_multiple(self, weights, nomsA, denomsA, nomsB, denomsB):
		self.A, self.B = ext.updatemult(weights, nomsA, denomsA, nomsB, denomsB)
