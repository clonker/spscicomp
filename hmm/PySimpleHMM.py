""" Python Hidden Markov Model kernel

A simple HMM implementation in pure python using numpy arrays but no fancy
vectorization or any other language bindings. This can be used for small models
and observation sequences. This class serves as demonstration what functions
any HMM class of this package should have.

Algorithms as proposed by L. Rabiner (see http://dx.doi.org/10.1109/5.18626 )

"""

import numpy as np
import utilities


class PySimpleHMM(object):
	def __init__(self, N, M, A, B, pi):
		self.N = N
		self.M = M
		self.A = A.copy()
		self.B = B.copy()
		self.pi = pi.copy()

	def forward(self, obs):
		T, N = len(obs), self.N
		alpha = np.zeros((T,N), dtype=np.float64)
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
		beta = np.zeros((T,N), dtype=np.float64)
		for i in range(N):
			beta[T-1,i] = 1
		for t in range(T-2, -1, -1):
			for i in range(N):
				beta[t,i] = 0.0
				for j in range(N):
					beta[t,i] += self.A[i,j] * beta[t+1,j] * self.B[j,obs[t+1]]
		return beta

	def computeGamma(self, alpha, beta):
		T, N = len(alpha), self.N
		gamma = np.zeros((T,N), dtype=np.float64)
		for t in range(T):
			sum = 0.0
			for i in range(N):
				gamma[t,i] = alpha[t,i]*beta[t,i]
				sum += gamma[t,i]
			for i in range(N):
				gamma[t,i] /= sum
		return gamma
		
	def computeXi(self, obs, alpha, beta):
		T, N = len(obs), self.N
		xi = np.zeros((T-1,N,N), dtype=np.float64)
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
		
	def computeNominatorA(self, obs, alpha, beta):
		T, N = len(obs), self.N
		xi = np.zeros((N,N), dtype=np.double)
		xi_t = np.zeros((N,N), dtype=np.double)
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
		
	def computeDenominatorA(self, gamma):
		denom = np.zeros((self.N), dtype=np.double)
		for t in range(len(gamma)-1):
			for i in range(self.N):
				denom[i] += gamma[t,i]
		return denom
	
	def computeNominatorB(self, obs, gamma):
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
		gamma = self.computeGamma(alpha, beta)
		nomA = self.computeNominatorA(obs, alpha, beta)
		denomA = self.computeDenominatorA(gamma)
		nomB = self.computeNominatorB(obs, gamma)
		denomB = denomA + gamma[T-1]
		return weight, nomA, denomA, nomB, denomB

	@profile
	def BaumWelch_multiple(self, obss, accuracy, maxiter):
		K, N, M = len(obss), self.N, self.M
		nomsA = np.zeros((K,N,N), dtype=np.double)
		denomsA = np.zeros((K,N), dtype=np.double)
		nomsB = np.zeros((K,N,M), dtype=np.double)
		denomsB = np.zeros((K,N), dtype=np.double)
		weights = np.zeros(K, dtype=np.double)
		
		old_eps = 0.0
		it = 0
		new_eps = accuracy+1

		while (abs(new_eps - old_eps) > accuracy and it < maxiter):
			for k in range(K):
				weights[k], nomsA[k], denomsA[k], nomsB[k], denomsB[k] = self.process_obs(obss[k])
				
			for i in range(N):
				nomA = np.zeros(N, dtype=np.double)
				denomA = 0.0
				for k in range(K):
					nomA += weights[k] * nomsA[k,i,:]
					denomA += weights[k] * denomsA[k,i]
				self.A[i,:] = nomA / denomA
			for i in range(N):
				nomB = np.zeros(M, dtype=np.double)
				denomB = 0.0
				for k in range(K):
					nomB += weights[k] * nomsB[k, i, :]
					denomB += weights[k] * denomsB[k, i]
				self.B[i,:] = nomB / denomB

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

#	@profile
	def BaumWelch(self, obs, accuracy, maxit):
		old_eps = 0.0
		it = 0
		T = len(obs)
		new_eps,alpha = self.forward(obs)
		while (abs(new_eps - old_eps) > accuracy and it < maxit):
			beta = self.backward(obs)
			gamma = self.computeGamma(alpha, beta)
			xi = self.computeXi(obs, alpha, beta)
			self.update(obs, gamma, xi)
			old_eps = new_eps
			new_eps, alpha = self.forward(obs)
			it += 1
		return (new_eps, it)
		
						
	def printModel(self):
		"""Print A, B and pi."""
		print 'A:\n', np.round(self.A, 2)
		print 'B:\n', np.round(self.B, 2)
		print 'pi:\n', np.round(self.pi, 2)

	def randomSequence(self, obsLength):
		"""Creates a random Sequence of length n on base of this model."""
		obs = np.empty(obsLength, dtype='int')
		current = utilities.random_by_dist(self.pi)
		for i in range(obsLength):
			obs[i]  = utilities.random_by_dist(self.B[current,:])
			current = utilities.random_by_dist(self.A[current])
		return obs
