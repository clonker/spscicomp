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
in: pi - startprobabilities
    A  - state transition probabilities
    B  - observation probabilities by state
    observationLength - length of observations
'''
def getObservationFromModel(pi, A, B, observationLength):
    observation = []
    # get the first urn
    currentState = getRandomIndexByProbability(pi)
    # create the observationLength observation
    for i in range(observationLength):
        # create current observation
        observation.append(getRandomIndexByProbability(B[currentState]))
        # get next urn
        currentState = getRandomIndexByProbability(A[currentState])

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

    originalTransitionMatrix =     np.loadtxt('testdata/urnModelA_' + testcase + '.dat')
    originalObservationProbs =     np.loadtxt('testdata/urnModelB_' + testcase + '.dat')
    originalInitialState =         np.loadtxt('testdata/urnModelPi_' + testcase + '.dat')
    
    #print len(getUrnObservation(initialState, transitionMatrix, observationProbs, observationCount))
    
    
    maxIterations = 100
    likelies = np.zeros(maxIterations-1)
    model = [transitionMatrix, observationProbs, initialState]
    #"""
    print 'transitionMatrix of model\n', originalTransitionMatrix
    print 'observationProbs of model\n', originalObservationProbs
    print 'initialState of model\n', originalInitialState

    print 'transitionMatrix\n', model[0]
    print 'observationProbs\n', model[1]
    print 'initialState\n', model[2]
    
    for iteration in range(1, maxIterations):
        #print iteration
        #print '---------------------------------------'
        observation = getObservationFromModel(originalInitialState, originalTransitionMatrix, originalObservationProbs, observationCount)
        model, scalingFactors = hmm.propagate(model, observation)
        likeli = hmm.logLikeli(scalingFactors)
        likelies[iteration-1] = likeli
    #"""
    print 'transitionMatrix\n', model[0]
    print 'observationProbs\n', model[1]
    print 'initialState\n', model[2]
    print 'loglikeli', likeli
    #"""
    plt.plot(likelies[1:])
    plt.show()
    