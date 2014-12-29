"""Hidden Markov Model kernel

The Hidden Markov model algorithm is an optimization procedure to find the
transition probability matrix of a 'hidden' system which describes a given
observation sequence.

Algorithms as proposed by L. Rabiner (see http://dx.doi.org/10.1109/5.18626 )

"""

from extension import hmm_ext as ext
import numpy as np
import matplotlib.pyplot as plt
import sys

class HiddenMarkovModel:
	"""HiddenMarkovModel

	This class represents a hidden markov model (hmm) and provides methods
	to create, alter and create Observation sequences of it. A hmm is
	characterized by:

	1) N, the number of states
	2) K, the number of distinct observation symbols
	3) A, the NxN transition matrix
	4) B, the KxN observation probabilty distribution
	5) pi, the initial state distribution

	"""
	# The constructor. There is no default.
	def __init__(self, N, K, A, B, pi):
		"""The constructor checks if all sizes are matching and so on."""
		self.N  = N
		self.K  = K
		if (A.shape == (N,N)) and (B.shape == (N,K)) and (pi.shape == (N,)):
			self.A  = A.copy()
			self.B  = B.copy()
			self.pi = pi.copy()
		else:
			raise ValueError('Dimension mismatch')

	def getModel(self):
		"""Returns [A,B,pi]."""
		return [self.A,self.B,self.pi]

	def printModel(self):
		"""Print A, B and pi."""
		print 'A\n', np.round(self.A, 3)
		print 'B\n', np.round(self.B, 3)
		print 'pi\n', np.round(self.pi, 3)

	def randomSequence(self, n):
		"""Creates a random Sequence of length n on base of this model."""
		A,B,pi = self.A,self.B,self.pi
		obs = np.empty(n, dtype='int')
		current = random_by_dist(pi)
		for i in range(n):
			obs[i]  = random_by_dist(B[current,:])
			current = random_by_dist(A[current])
		return obs

#	@profile
	def BaumWelch(self, observation, epsilon, maxIter, verbose=False):
		"""Optimize the given model.

		Use the Hidden Markov Model algorithm to optimize the given
		model. The procedure ends if the likelihood stabilizes up to
		a given constant 'epsilon' OR the maximum number of iterations
		'maxIter' is reached. 
		Return the optimized model and an array of the likelihoods 
		recorded along the process.

		The model is a tupel containing the following quantities:
		model = (transitionMatrix, observationProbs, initialState)

		:observation: sequence of observed states
		:epsilon:     desired stability as ending criterion
		:maxIter:     number how often the model is updated
		:verbose:     print additional information one the fly

		Return an array of logarithmized likelihoods during optimization.

		"""
		T = len(observation)
		N = len(self.A)
		gamma   = np.zeros((T,N), dtype=np.double)
		xi      = np.zeros((T-1,N,N), dtype=np.double)
		alpha   = np.zeros((T,N), dtype=np.double)
		beta    = np.zeros((T,N), dtype=np.double)
		scaling = np.zeros(T, dtype=np.float64)

		likelies = np.zeros(maxIter)
		i = 0
		# make sure that the loop is entered the first time
		oldLike = 0.
		newLike = 10.
		if (verbose):
			print 'Epsilon:', epsilon
			print 'Iteration:', i, '/', maxIter
			print 'LogLikelihood:', newLike
			print 'Difference Likelihood:', abs(oldLike - newLike)
		A,B,pi = self.A.copy(), self.B.copy(), self.pi.copy()
		while ( (i < maxIter) and (abs(oldLike - newLike) > epsilon) ):
			if (verbose):
				sys.stdout.write(3 * "\033[F") # delete last 3 lines
				print 'Iteration:', i, '/', maxIter
				print 'LogLikelihood:', newLike
				print 'Difference Likelihood:', abs(oldLike - newLike)

			likelies[i] = ext.forward(alpha, scaling, A, B, pi, observation)
			ext.backward(beta, scaling, A, B, pi, observation)
			ext.compute_gamma(alpha, beta, gamma)
			ext.compute_xi(A, B, pi, observation, alpha, beta, xi)	
			ext.update(A, B, pi, observation, gamma, xi)

			newLike        = likelies[i]
			if (i!=0):
				oldLike    = likelies[i-1]
			else:
				oldLike    = 0.
			i += 1
		self.A,self.B,self.pi = A,B,pi
		print 'Terminated after ', i, ' iterations'
		return likelies[0:i]


# @profile
def update_model(A, B, pi, alpha, beta, obs, gamma, xi):
	"""Update a model based on given forward/backward coefficients.

	This is a non-parallel version of updating a given model with only one
	given observation sequence `observation'. Also this method does not
	calculate the forward coefficients `alpha' nor the backward coefficients
	`beta'. The applied formula holds for scaled or non-scaled coefficients.
	Calculate `alpha' and `beta' with methods like `forward' and `backward'.

	There are some preassumtion when using this function.

	"""
	return ext.update(A, B, pi, alpha, beta, obs, gamma, xi)

# @profile
def forward(A, B, pi, observation, alpha, scaling):
	"""Generate the forward coeffcients and scaling factors.

	The forward coefficients are represented as a matrix of T rows and N
	columns, where T is the observation length and N is the number of hidden
	states. All forward coefficients in one row are normalized to 1. The
	factors to normalize them are saved in a vector c and are later needed
	to calculate the likelihood.

	"""
	return ext.forward(alpha, scaling, A, B, pi, observation)


# @profile
def backward(A, B, pi, observation, scaling, beta):
	"""Generate the backward coefficients with the scaling factors of the forward coefficients."""
	return ext.backward(beta, scaling, A, B, pi, observation)

def random_by_dist(distribution):
	x = np.random.random();
	for n in range(len(distribution)):
		if x < (distribution[n]):
			return n;
		else:
			x -= distribution[n];
	#print 'Reached exceptional state in random_by_dist()'
	return n

def compare(model1, model2, obsLength):
	"""Quantify the similarity of two models, based on an observation 
	sequence of model2."""
	A1  = model1.A.copy()
	A2  = model2.A.copy()
	B1  = model1.B.copy()
	B2  = model2.B.copy()
	pi1 = model1.pi.copy()
	pi2 = model2.pi.copy()
	alpha1 = np.zeros((obsLength, len(A1)), dtype=np.float64)
	alpha2 = np.zeros((obsLength, len(A2)), dtype=np.float64)
	scaling1 = np.zeros(obsLength, dtype=np.float64)
	scaling2 = np.zeros(obsLength, dtype=np.float64)
	observation = model2.randomSequence(obsLength)
	forward(A1, B1, pi1, observation, alpha1, scaling1)
	forward(A2, B2, pi2, observation, alpha2, scaling2)
	result = -np.sum(np.log(scaling1)) + np.sum(np.log(scaling2))
	return result / float(obsLength)

def similarity(model1, model2, obsLength):
	"""Give a symmetric realisation of the similarity.

	Average the unsymmetric similarities compare(model1, model2)
	and compare(model2, model1).

	"""
	com1 = compare(model1, model2, obsLength)
	com2 = compare(model2, model1, obsLength)
	return 0.5*(com1 + com2)

