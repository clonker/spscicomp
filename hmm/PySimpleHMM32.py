import numpy as np


class PySimpleHMM32(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		self.N = N
		self.M = M
		self.A = np.asarray(A.copy(), dtype=np.float32)
		self.B = np.asarray(B.copy(), dtype=np.float32)
		self.pi = np.asarray(pi.copy(), dtype=np.float32)

	def forward(self, obs):
		T, N = len(obs), self.N
		alpha = np.zeros((T,N), dtype=np.float32)
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
		beta = np.zeros((T,N), dtype=np.float32)
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
		gamma = np.zeros((T,N), dtype=np.float32)
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
		xi = np.zeros((T-1,N,N), dtype=np.float32)
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
		xi = np.zeros((N,N), dtype=np.float32)
		xi_t = np.zeros((N,N), dtype=np.float32)
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
		denom = np.zeros((self.N), dtype=np.float32)
		for t in range(len(gamma)-1):
			for i in range(self.N):
				denom[i] += gamma[t,i]
		return denom


#	@profile
	def BaumWelch_multiple(self, obss, accuracy, maxiter):
		K, N, M = len(obss), self.N, self.M
		nomsA = np.zeros((K,N,N), dtype=np.float32)
		denomsA = np.zeros((K,N), dtype=np.float32)
		nomsB = np.zeros((K,N,M), dtype=np.float32)
		denomsB = np.zeros((K,N), dtype=np.float32)
		weights = np.zeros(K, dtype=np.float32)
		
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
