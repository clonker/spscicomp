#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
# import hmm.py to have access to Hidden Markov Model kernel
import hmm

sample1 = np.loadtxt('testdata/sample1.dat')
shortSample = sample1[0:100]

testcase = '1'
transitionMatrix = 	np.loadtxt('testdata/startmodelA_' + testcase + '.dat')
observationProbs = 	np.loadtxt('testdata/startmodelB_' + testcase + '.dat')
initialState = 		np.loadtxt('testdata/startmodelPi_' + testcase + '.dat')

model = [transitionMatrix, observationProbs, initialState]
model, likelies = hmm.optimize(model, shortSample, 700, verbose=True)

plt.plot(likelies)
plt.show()
