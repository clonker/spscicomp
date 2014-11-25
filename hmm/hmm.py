"""Hidden Markov Model kernel

The Hidden Markov model algorithm is an optimization procedure to find the
transition probability matrix of a 'hidden' system which describes a given
observation sequence.
As proposed by L. Rabiner (see http://dx.doi.org/10.1109/5.18626 )

"""

from extension import hmm_ext as ext
import numpy as np
import matplotlib.pyplot as plt

def optimize(model, observation, maxIterations, verbose=False):
	"""Optimize the given model.

	Use the Hidden Markov Model algorithm to optimize the given
	model for 'maxIterations' times. Return the optimized model
	and an array of the likelihoods recorded along the process.

	The model is a tupel containing the following quantities:
	model = (transitionMatrix, observationProbs, initialState)

	:param1: model parameters subject to optimization
	:param2: sequence of observed states
	:param3: number how often the model is updated
	:param4: decide if model and likelihood will be printed
	:return: optimized model, array of logarithmized likelihoods during
	         optimization

	"""
	likelies = np.zeros(maxIterations-1)
	for iteration in range(1, maxIterations):
		alpha, scaling = forward(model, observation)
		beta           = backward(model, observation, scaling)
		model          = update_model(model, alpha, beta, observation)
		likeli         = -np.sum(np.log(scaling))
		likelies[iteration-1] = likeli

	if verbose:
		print 'transitionMatrix\n', np.round(model[0], 3)
		print 'observationProbs\n', np.round(model[1], 3)
		print 'initialState\n', np.round(model[2], 3)
		print 'loglikeli', likeli

	return (model, likelies)

def update_model(model, alpha, beta, observation):
	"""Update a model based on given forward/backward coefficients.

	This is a non-parallel version of updating a given model with only one
	given observation sequence `observation'. Also this method does not
	calculate the forward coefficients `alpha' nor the backward coefficients
	`beta'. The applied formula holds for scaled or non-scaled coefficients.
	Calculate `alpha' and `beta' with methods like `forward' and `backward'.

	There are some preassumtion when using this function.

	1) The initial `model' is given as an array of the form [A,B,pi].
	N is the number of hidden states. K the number different observations.
	   - `A' is a NxN-matrix of transition probabilities between states
	   - `B' is a KxN-matrix of observation probabilities per state
	   - `pi' is an one-dimensional array length N of the initial state
	     probability distribution
	2) There are some restrictions for `alpha', `beta' and `observation'
	Let T be the amount of observations, s.t. len(observation) = T
	   - `alpha' is a TxN-matrix
	   - `beta' is a TxN-matrix
	   - `observation' is an one-dimensional array of length T and takes a
	     maximum of K different symbols
	3) It returns a model of the form [A,B,pi].

	"""
	# get abbreviations
	A,B,pi,O = model[0],model[1],model[2],observation

	# this is used for pi and B
	gamma = alpha*beta # gamma(t,i) = alpha(t,i)*beta(t,i)
	gamma = (gamma.T / np.sum(gamma, axis=1)).T # normalize each row
	# update initial state
	pi = gamma[0]

	# update transitition matrix
	T = len(O)
	A *= np.dot( alpha[0:T-1].T , beta[1:T]*B[O[1:T]] )
	A = (A.T / np.sum(A, axis=1)).T # normalize each row

	# update state probabilities
	for k in range(0, len(B)):
		# TODO this possibly adds alot (T-many) zeros
		B[k] = np.sum( ((O == k) * gamma.T).T, axis=0 )
	B /= np.sum(B, axis=0) # normalize each column

	return [A,B,pi]

def forward(model, observation):
	"""Generate the forward coeffcients and scaling factors.

	The forward coefficients are represented as a matrix of T rows and N
	columns, where T is the observation length and N is the number of hidden
	states. All forward coefficients in one row are normalized to 1. The
	factors to normalize them are saved in a vector c and are later needed
	to calculate the likelihood.

	"""
	# get model parameter
	A,B,pi,O = model[0],model[1],model[2],observation
	T,N = len(O), len(A)

	# allocate memory
	alpha = np.zeros((T,N), dtype='double') # array of forward coefficients
	c = np.zeros(T, dtype='double')         # scaling factors
	if (T == 0):
		return (alpha,c)

	return ext.forward(alpha, c, A, B, pi, observation)

"""
	# Initialization for t=0:
	alpha[0] = pi*B[O[0]];
	c[0] = 1. / np.sum(alpha[0])
	# rescale alpha
	alpha[0] *= c[0]

	# Induction for 0 < t < T:
	for t in range(1,T):
		alpha[t] = np.dot(alpha[t-1],A)*B[O[t]]
		c[t] = 1./np.sum(alpha[t])
		# rescale alpha
		alpha[t] *= c[t]
	return (alpha, c)
"""

def backward(model, observation, scaling):
	"""Generate the backward coefficients with the scaling factors of the
	forward coefficients.

	"""
	# get model parameter
	A,B,pi,O,c = model[0],model[1],model[2],observation,scaling
	T, N = len(O), len(A)

	# allocate memory
	beta = np.zeros((T,N))
	if T == 0:
		return []

	return ext.backward(beta, scaling, A, B, pi, observation)
"""
	# Initialization for t=T:
	beta[T-1] = c[T-1] # rescale betas with factors from forward calculation

	# Induction for T > t > 0:
	for t in range(T-2,-1,-1):
		beta[t] = c[t]*np.dot(A, B[O[t+1]]*beta[t+1])

	return beta
"""

