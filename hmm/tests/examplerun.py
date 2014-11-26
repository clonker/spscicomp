#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
# import hmm.py to have access to Hidden Markov Model kernel
from hmm import *

sample1 = np.loadtxt('data/sample1.dat', dtype='int')
sample1 = sample1[0:100]

testcase = '1'
A = np.loadtxt('data/startmodelA_' + testcase + '.dat')
B = np.loadtxt('data/startmodelB_' + testcase + '.dat').T
pi = np.loadtxt('data/startmodelPi_' + testcase + '.dat')

hmm = HiddenMarkovModel(len(A), len(B), A, B, pi)
A,B,pi,likelies = hmm.optimize(sample1, 1100)

print 'A =\n',np.round(A,3)
print 'B =\n', np.round(B,3)
print 'pi =\n', np.round(pi,3)


plt.plot(likelies[1:], color='black')
plt.xlabel('number of iterations', fontsize=14)
plt.ylabel(r'$\ln\, P(O|\lambda) $', fontsize=16)
plt.show()

