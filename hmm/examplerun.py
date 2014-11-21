#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
# import hmm.py to have access to Hidden Markov Model kernel
import hmm

sample1 = np.loadtxt('testdata/sample1.dat')
shortSample = sample1[0:100]

testcase = '2'
transitionMatrix = 	np.loadtxt('testdata/startmodelA_' + testcase + '.dat')
observationProbs = 	np.loadtxt('testdata/startmodelB_' + testcase + '.dat')
initialState = 		np.loadtxt('testdata/startmodelPi_' + testcase + '.dat')

model = [transitionMatrix, observationProbs, initialState]
model, likelies = hmm.optimize(model, shortSample, 1100, verbose=True)

plt.plot(likelies[1:], color='black')
plt.xlabel('number of iterations', fontsize=14)
plt.ylabel(r'$\ln\, P(O|\lambda) $', fontsize=16)
plt.show()
