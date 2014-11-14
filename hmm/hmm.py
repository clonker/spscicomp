#!/bin/python
"""
This is the documentation
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

# performs the update scheme once
def propagate(model, observation):
		transitionMatrix 	= model[0]
		observationProbs	= model[1]
		initialState 		= model[2]
		
		newTransitionMatrix	= np.zeros(transitionMatrix.shape)
		newObservationProbs	= np.zeros(observationProbs.shape)
		newInitialState		= np.zeros(initialState.shape)
		
		forwardCoeffs, scalingFactors = scaledForwardCoeffs(model, observation)
		backwardCoeffs = scaledBackwardCoeffs(model, observation, scalingFactors)
		
		for i in range(0, len(newInitialState)):
			newInitialState[i] = transitionFrom(0, i, forwardCoeffs, backwardCoeffs)

		for i in range(0, len(newTransitionMatrix)):
			for j in range(0, len(newTransitionMatrix[i])):
				enumerator, denominator = 0., 0.
				for t in range(0, len(observation)-1):
					enumerator  += transitionToFrom(t, i, j, forwardCoeffs, backwardCoeffs, transitionMatrix, observationProbs, observation)
					denominator += transitionFrom(t, i, forwardCoeffs, backwardCoeffs)
				newTransitionMatrix[i,j] = enumerator / denominator
		
		for i in range(0, len(newObservationProbs)):
			for k in range(0, len(newObservationProbs[i])):
				enumerator, denominator = 0., 0.
				for t in range(0, len(observation)):
					if (observation[t] == k):
						enumerator += transitionFrom(t, i, forwardCoeffs, backwardCoeffs)
					denominator += transitionFrom(t,i,forwardCoeffs, backwardCoeffs)
				newObservationProbs[i,k] = enumerator / denominator
		
		newModel = [newTransitionMatrix, newObservationProbs, newInitialState]
		return newModel
		
# this computes the gamma
def transitionFrom(t, i, alpha, beta):
	norm = 0.
	for j in range(0,len(alpha[0])):
		norm += alpha[t,j] * beta[t,j]
	return alpha[t,i] * beta[t,i] / norm

# this computes the xi
def transitionToFrom(t, i, j, alpha, beta, transitionMatrix, observationProbs, observation):
	A = transitionMatrix
	B = observationProbs
	norm = 0.
	for k in range(0,len(A)):
		for l in range(0,len(A)):
			norm += alpha[t,k] * A[k,l] * B[l, observation[t+1]] * beta[t+1,l]
	return alpha[t,i] * A[i,j] * B[j, observation[t+1]] * beta[t+1,j] / norm
	
# generate the forward coeffcients and scaling factors
def scaledForwardCoeffs(model, observation):
	A = model[0]	# transition matrix
	B = model[1]	# observation probabilites
	pi = model[2]	# initial state
	T = len(observation)
	N = len(A)
	alpha = np.zeros((T,N))	# array of forward coefficients
	c = np.zeros(T)	# scaling factors
	
	# Initialization for t=0:
	for i in range(0,N):
		alpha[0,i] = pi[i] * B[i, observation[0]]
	c[0] = 1./sum(alpha[0,:])
	alpha[0,:] *= c[0]	# rescale alphas by factor c
	
	# Induction for 0 < t < T:
	for t in range(1,T):
		for i in range(0,N):
			alpha[t,i] = alphaCoeff(t, i, alpha[t-1,:], A, B, observation)
		c[t] = 1./sum(alpha[t,:])
		alpha[t,:] *= c[t]
	
	return (alpha, c)

def alphaCoeff(t, i, preAlpha, A, B, observation):
	result = 0.
	for j in range(0, len(A)):
		result += preAlpha[j] * A[j,i]
	return result*B[i, observation[t]]

# generate the backward coefficients with the scalingFactors of the forward coefficients
def scaledBackwardCoeffs(model, observation, scalingFactors):
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
	beta[T-1,:] *= c[T-1]	# rescale betas with factors from forward calculation
	
	# Induction for T > t > 0:
	for t in range(T-2,-1,-1):
		for i in range(0,N):
			beta[t,i] = betaCoeff(t, i, beta[t+1,:], A, B, observation)
		beta[t,:] *= c[t]
	
	return beta

def betaCoeff(t, i, preBeta, A, B, observation):
	result = 0.
	for j in range(0,len(A)):
		result += A[i,j] * B[j, observation[t+1]] * preBeta[j]
	return result

# calculate the logarithm of the likelihood, simply as the
# sum of the logarithmized scaling factors
def logLikeli(scalingFactors):
	result = 0.
	for i in range(0,len(scalingFactors)):
		result += np.log(scalingFactors[i])
	return -1. * result

"""-----------------------------------------------------------------------------------------"""

sample1 = np.loadtxt('testdata/sample1.dat')
shortSample = sample1[0:100]

testcase = '1'
transitionMatrix = 	np.loadtxt('testdata/startmodelA_' + testcase + '.dat')
observationProbs = 	np.loadtxt('testdata/startmodelB_' + testcase + '.dat')
initialState = 		np.loadtxt('testdata/startmodelPi_' + testcase + '.dat')

maxIterations = 700
likelies = np.zeros(maxIterations)
model = [transitionMatrix, observationProbs, initialState]
alpha, scalingFactors = scaledForwardCoeffs(model, shortSample)
likeli = logLikeli(scalingFactors)
#"""
print 'transitionMatrix\n', model[0]
print 'observationProbs\n', model[1]
print 'initialState\n', model[2]
print 'loglikeli', likeli
#"""
likelies[0] = likeli

for iteration in range(1, maxIterations):
	#print iteration
	#print '---------------------------------------'
	model = propagate(model, shortSample)
	alpha, scalingFactors = scaledForwardCoeffs(model, shortSample)
	likeli = logLikeli(scalingFactors)
	likelies[iteration] = likeli
#"""
print 'transitionMatrix\n', model[0]
print 'observationProbs\n', model[1]
print 'initialState\n', model[2]
print 'loglikeli', likeli
#"""
plt.plot(likelies[1:])
plt.show()
