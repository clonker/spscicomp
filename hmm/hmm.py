"""Hidden Markov Model kernel

The Hidden Markov model algorithm is an optimization procedure to find the
transition probability matrix of a 'hidden' system which describes a given
observation sequence.

Algorithms as proposed by L. Rabiner (see http://dx.doi.org/10.1109/5.18626 )

"""

from extension import hmm_ext as ext
import numpy as np
import matplotlib.pyplot as plt

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
		if (A.shape == (N,N)) and (B.shape == (K,N)) and (pi.shape == (N,)):
			self.A  = A.copy()
			self.B  = B.copy()
			self.pi = pi.copy()
		else:
			raise ValueError('Dimension mismatch')

	def getModel(self):
		"""Returns [A,B,pi]."""
		return [self.A,self.B,self.pi]

	def randomSequence(self, n):
		"""Creates a random Sequence of length n on base of this model."""
		A,B,pi = self.A,self.B.T,self.pi
		obs = np.empty(n, dtype='int')
		current = random_by_dist(pi)
		for i in range(n):
			obs[i]  = random_by_dist(B[current])
			current = random_by_dist(A[current])
		return obs

#	@profile
	def optimize(self, observation, maxIter):
		"""Optimize the given model.

		Use the Hidden Markov Model algorithm to optimize the given
		model for 'maxIterations' times. Return the optimized model
		and an array of the likelihoods recorded along the process.

		The model is a tupel containing the following quantities:
		model = (transitionMatrix, observationProbs, initialState)

		:observation: sequence of observed states
		:maxIter:     number how often the model is updated

		returns an array of logarithmized likelihoods during optimization.

		"""
		A,B,pi = self.A.copy(), self.B.copy(), self.pi.copy()
		likelies = np.zeros(maxIter)
		for i in range(0, maxIter):
			alpha, scaling = forward(A, B, pi, observation)
			beta           = backward(A, B, pi, observation, scaling)
			A,B,pi         = update_model(A, B, pi, alpha, beta, observation)
			likelies[i]    = -np.sum(np.log(scaling))

		return (A,B,pi,likelies)


# @profile
def update_model(A, B, pi, alpha, beta, obs):
	"""Update a model based on given forward/backward coefficients.

	This is a non-parallel version of updating a given model with only one
	given observation sequence `observation'. Also this method does not
	calculate the forward coefficients `alpha' nor the backward coefficients
	`beta'. The applied formula holds for scaled or non-scaled coefficients.
	Calculate `alpha' and `beta' with methods like `forward' and `backward'.

	There are some preassumtion when using this function.

	"""
	return ext.update_model(A, B, pi, alpha, beta, obs)

# @profile
def forward(A, B, pi, observation):
	"""Generate the forward coeffcients and scaling factors.

	The forward coefficients are represented as a matrix of T rows and N
	columns, where T is the observation length and N is the number of hidden
	states. All forward coefficients in one row are normalized to 1. The
	factors to normalize them are saved in a vector c and are later needed
	to calculate the likelihood.

	"""
	# allocate memory
	T,N = len(observation),len(A)
	alpha = np.zeros((T,N), dtype='double') # array of forward coefficients
	scaling = np.zeros(T, dtype='double')              # scaling factors
	if (T == 0):
		return (alpha,scaling)

	return ext.forward(alpha, scaling, A, B, pi, observation)


# @profile
def backward(A, B, pi, observation, scaling):
	"""Generate the backward coefficients with the scaling factors of the forward coefficients."""
	# allocate memory
	T,N = len(observation),len(A)
	beta = np.zeros((T,N))
	if T == 0:
		return []

	return ext.backward(beta, scaling, A, B, pi, observation)

def random_by_dist(distribution):
	x = np.random.random();
	for n in range(len(distribution)):
		if x < distribution[n]:
			return n;
		else:
			x -= distribution[n];

