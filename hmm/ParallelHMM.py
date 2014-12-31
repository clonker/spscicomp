import numpy as np
from PySimpleHMM import *
from pathos.multiprocessing import ProcessingPool as Pool

class ParallelHMM(PySimpleHMM):
	def __init__(self, N, M, A, B, pi):
		super(ParallelHMM, self).__init__(N, M, A, B, pi)

	@profile
	def BaumWelch_multiple(self, obss, accuracy, maxiter):
		K, N, M = len(obss), self.N, self.M
		nomsA = np.zeros((K,N,N), dtype=np.double)
		denomsA = np.zeros((K,N), dtype=np.double)
		nomsB = np.zeros((K,N,M), dtype=np.double)
		denomsB = np.zeros((K,N), dtype=np.double)
		weights = np.zeros(K, dtype=np.double)
		
		# creating Thread pool here
		# TODO have to determine best number of threads
		pool = Pool(8)
		
		old_eps = 0.0
		it = 0
		new_eps = accuracy+1

		while (abs(new_eps - old_eps) > accuracy and it < maxiter):
			
			values = pool.map(ParallelHMM.process_obs, [self]*len(obss), obss)
			[weights, nomsA, denomsA, nomsB, denomsB] = [np.array(t) for t in zip(*values)]
			
			# update HMM
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
