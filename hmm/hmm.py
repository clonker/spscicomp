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
		model, scalingFactors = propagate(model, observation)
		likeli = logLikeli(scalingFactors)
		likelies[iteration-1] = likeli
	if verbose:
		print 'transitionMatrix\n', model[0]
		print 'observationProbs\n', model[1]
		print 'initialState\n', model[2]
		print 'loglikeli', likeli
	return (model, likelies)	

def propagate(model, observation):
	"""Perform the update scheme once.
	
	Use the update formulas given in the Rabiner paper. Note that this 
	function returns the scalingFactors of the 'old' model.
	
	"""
	transitionMatrix 	= model[0]
	observationProbs	= model[1]
	initialState 		= model[2]
	
	newTransitionMatrix	= np.zeros(transitionMatrix.shape)
	newObservationProbs	= np.zeros(observationProbs.shape)
	newInitialState		= np.zeros(initialState.shape)
	
	forwardCoeffs, scalingFactors = scaledForwardCoeffs(model, observation)
	backwardCoeffs = scaledBackwardCoeffs(model, observation, scalingFactors)
	
	for i in range(0, len(newInitialState)):
		newInitialState[i] = transitionFrom(0, i, forwardCoeffs, 
			backwardCoeffs)

	for i in range(0, len(newTransitionMatrix)):
		for j in range(0, len(newTransitionMatrix[i])):
			enumerator, denominator = 0., 0.
			for t in range(0, len(observation)-1):
				enumerator  += transitionToFrom(t, i, j, forwardCoeffs, 
					backwardCoeffs, transitionMatrix, observationProbs, 
					observation)
				denominator += transitionFrom(t, i, forwardCoeffs, 
					backwardCoeffs)
			newTransitionMatrix[i,j] = enumerator / denominator
	
	for i in range(0, len(newObservationProbs)):
		for k in range(0, len(newObservationProbs[i])):
			enumerator, denominator = 0., 0.
			for t in range(0, len(observation)):
				if (observation[t] == k):
					enumerator += transitionFrom(t, i, forwardCoeffs, 
						backwardCoeffs)
				denominator += transitionFrom(t,i,forwardCoeffs, 
					backwardCoeffs)
			newObservationProbs[i,k] = enumerator / denominator
	
	newModel = (newTransitionMatrix, newObservationProbs, newInitialState)
	return (newModel, scalingFactors)
		
def transitionFrom(t, i, alpha, beta):
	"""Compute the probability of being in state i at time t."""
	norm = 0.
	for j in range(0,len(alpha[0])):
		norm += alpha[t,j] * beta[t,j]
	return alpha[t,i] * beta[t,i] / norm

def transitionToFrom(t, i, j, alpha, beta, transitionMatrix, 
	observationProbs, observation):
	"""Compute the probability of being in state i at time t and going to 
	state j."""
	A = transitionMatrix
	B = observationProbs
	norm = 0.
	for k in range(0,len(A)):
		for l in range(0,len(A)):
			norm += alpha[t,k] * A[k,l] * B[l, observation[t+1]] * beta[t+1,l]
	return alpha[t,i] * A[i,j] * B[j, observation[t+1]] * beta[t+1,j] / norm
	
def scaledForwardCoeffs(model, observation):
	"""Generate the forward coeffcients and scaling factors.
	
	The forward coefficients are represented as a matrix of T rows and N
	columns, where T is the observation length and N is the number of hidden
	states. All forward coefficients in one row are normalized to 1. The 
	factors to normalize them are saved in a vector c and are later needed to
	calculate the likelihood.
	
	"""
	A = model[0]	# transition matrix
	B = model[1]	# observation probabilites
	pi = model[2]	# initial state
	T = len(observation)
	N = len(A)
	alpha = np.zeros((T,N))	# array of forward coefficients
	c = np.zeros(T)	# scaling factors
	
	# Initialization for t=0:
	for i in range(0,N):
		alpha[0, i] = pi[i] * B[i, observation[0]]
	c[0] = 1./sum(alpha[0,:])
	alpha[0,:] *= c[0]	# rescale alphas by factor c
	
	# Induction for 0 < t < T:
	for t in range(1,T):
		for i in range(0,N):
			alpha[t, i] = alphaCoeff(t, i, alpha[t-1, :], A, B, observation)
		c[t] = 1./sum(alpha[t, :])
		alpha[t, :] *= c[t]
	
	return (alpha, c)

def alphaCoeff(t, i, preAlpha, A, B, observation):
	"""Calculate one single entry in the forward coefficient matrix."""
	result = 0.
	for j in range(0, len(A)):
		result += preAlpha[j] * A[j, i]
	return result*B[i, observation[t]]

def scaledBackwardCoeffs(model, observation, scalingFactors):
	"""Generate the backward coefficients with the scaling factors of the 
	forward coefficients."""
	A = model[0]	# transition matrix
	B = model[1]	# observation probabilites
	pi = model[2]	# initial state
	T = len(observation)
	N = len(A)
	beta = np.zeros((T,N))	# array of backward coefficients
	c = scalingFactors	# scaling factors
	
	# Initialization for t=0:
	for i in range(0,N):
		beta[T-1,i] = 1.
	beta[T-1,:] *= c[T-1] # rescale betas with factors from forward calculation
	
	# Induction for T > t > 0:
	for t in range(T-2,-1,-1):
		for i in range(0,N):
			beta[t,i] = betaCoeff(t, i, beta[t+1,:], A, B, observation)
		beta[t,:] *= c[t]
	
	return beta

def betaCoeff(t, i, preBeta, A, B, observation):
	"""Calculate one single entry in the backward coefficient matrix."""
	result = 0.
	for j in range(0,len(A)):
		result += A[i,j] * B[j, observation[t+1]] * preBeta[j]
	return result

def logLikeli(scalingFactors):
	"""Calculate the logarithm of the likelihood, simply as the sum of the 
	logarithmized scaling factors."""
	result = 0.
	for i in range(0,len(scalingFactors)):
		result += np.log(scalingFactors[i])
	return -1. * result
