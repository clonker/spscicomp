'''
Created on 14.11.2014

@author: Duck Dodgers


Urnenmodell:
3 Urnen

Urnenstartwahrscheinlichkeit:
0.1 0.3 0.6

Wahrscheinlichkeit Urnenwechsel

0.7 0.2 0.1
0.4 0.5 0.1
0.2 0.3 0.5

Urne 1
20 rote
10 gelbe
50 blaue
20 gruene

Urne 2
10 rote
30 gelbe
20 blaue
40 gruene

Urne 3
30 rote
30 gelbe
10 blaue
20 gruene

'''

import numpy as np
import matplotlib.pyplot as plt
import hmm as hmm

'''

'''
def getObservationFromModel(pi, A, B, observationLength):
    observation = []
    # get the first urn
    currentUrn = getRandomIndexByProbability(pi)
    # create the observationLength observation
    for i in range(observationLength):
        # create current observation
        observation.append(getRandomIndexByProbability(B[currentUrn]))
        # get next urn
        currentUrn = getRandomIndexByProbability(A[currentUrn])

    return observation

'''
in: array of probabilities. Sum have to be 1
out: index randomly chosen by probability

creates a random number between 0..1 then chooses the index by the probabilities in the array
'''
def getRandomIndexByProbability(probs):
    r = np.random.random()
    for i in range(len(probs)):
        if r<probs[i]:
            return i
        else:
            r -= probs[i]



if __name__ == '__main__':
    testcase = '1'
    observationCount = 100
    transitionMatrix =     np.loadtxt('testdata/startUrnModelA_' + testcase + '.dat')
    observationProbs =     np.loadtxt('testdata/startUrnModelB_' + testcase + '.dat')
    initialState =         np.loadtxt('testdata/startUrnModelPi_' + testcase + '.dat')
    
    #print len(getUrnObservation(initialState, transitionMatrix, observationProbs, observationCount))
    
    
    maxIterations = 1000
    likelies = np.zeros(maxIterations)
    model = [transitionMatrix, observationProbs, initialState]
    alpha, scalingFactors = hmm.scaledForwardCoeffs(model, getObservationFromModel(initialState, transitionMatrix, observationProbs, observationCount))
    likeli = hmm.logLikeli(scalingFactors)
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
        observation = getObservationFromModel(initialState, transitionMatrix, observationProbs, observationCount)
        model = hmm.propagate(model, observation)
        alpha, scalingFactors = hmm.scaledForwardCoeffs(model, observation)
        likeli = hmm.logLikeli(scalingFactors)
        likelies[iteration] = likeli
    #"""
    print 'transitionMatrix\n', model[0]
    print 'observationProbs\n', model[1]
    print 'initialState\n', model[2]
    print 'loglikeli', likeli
    #"""
    plt.plot(likelies[1:])
    plt.show()
    