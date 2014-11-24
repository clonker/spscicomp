"""Hidden Markov Model kernel

The Hidden Markov model algorithm is an optimization procedure to find the
transition probability matrix of a 'hidden' system which describes a given
observation sequence.
As proposed by L. Rabiner (see http://dx.doi.org/10.1109/5.18626 )

"""

# TODO chris documentation Pydoc
# TODO chris split 'kernel' from exampleruns
# TODO tobias add ending criterion
# TODO honglei how to construct initial condition
# TODO maikel automatized TESTS

# TODO multiple observations
# TODO parallelize over different observation sequences
# TODO do in C

import numpy as np
import matplotlib.pyplot as plt

def optimize(model, observation, maxIterations, verbose=False):
	"""
	Optimize the given model.

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
		likeli         = logLikeli(scaling)
		likelies[iteration-1] = likeli

	if verbose:
		print 'transitionMatrix\n', model[0]
		print 'observationProbs\n', model[1]
		print 'initialState\n', model[2]
		print 'loglikeli', likeli

	return (model, likelies)

def update_model(model, alpha, beta, O):
	A,B,pi = model[0],model[1],model[2]

	gamma = alpha*beta
	gamma = (gamma.T / np.sum(gamma, axis=1)).T
	# update initial state
	pi = alpha[0]*beta[0] / np.dot(alpha[0],beta[0])

	# update transitition matrix
	T = len(O)
	A *= np.dot( alpha[0:T-1].T , beta[1:T]*B[O[1:T]] )
	A = (A.T / np.sum(A, axis=1)).T # normalize each row

	# update state probabilities
	for k in range(0, len(B)):
		B[k] = np.sum( ((O == k) * gamma.T).T, axis=0 )
	B /= np.sum(B, axis=0) # normalize each column

	return [A,B,pi]

def forward(model, observation):
	"""Generate the forward coeffcients and scaling factors.

	The forward coefficients are represented as a matrix of T rows and N
	columns, where T is the observation length and N is the number of hidden
	states. All forward coefficients in one row are normalized to 1. The
	factors to normalize them are saved in a vector c and are later needed to
	calculate the likelihood.

	"""
	# get model parameter
	A,B,pi,O = model[0],model[1],model[2],observation
	T,N = len(O), len(A)

	# allocate memory
	alpha = np.zeros((T,N))	# array of forward coefficients
	c = np.zeros(T)	        # scaling factors
	if (T == 0):
		return (alpha,c)

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

	# Initialization for t=T:
	beta[T-1] = c[T-1] # rescale betas with factors from forward calculation

	# Induction for T > t > 0:
	for t in range(T-2,-1,-1):
		beta[t] = c[t]*np.dot(A, B[O[t+1]]*beta[t+1])

	return beta

def logLikeli(scalingFactors):
	"""Calculate the logarithm of the likelihood, simply as the sum of the
	logarithmized scaling factors."""
	result = 0.
	for i in range(0,len(scalingFactors)):
		result += np.log(scalingFactors[i])
	return -1. * result
