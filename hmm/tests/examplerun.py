#!/bin/python

import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
# import hmm.py to have access to Hidden Markov Model kernel
from PySimpleHMM import *
from PyHMM import *
from CHMM import *
from NumPyHMM import *
from FortranHMM import *

obs_list = [
	np.loadtxt('data/t2.5.dat', dtype='int'),
	np.loadtxt('data/t1.100.dat', dtype='int'),
	np.loadtxt('data/t1.1000.dat', dtype='int'),
	np.loadtxt('data/t1.2000.dat', dtype='int'),
	np.loadtxt('data/t1.10000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int')
]

A = np.loadtxt('data/test_A.hmm')
B = np.loadtxt('data/test_B.hmm')
pi = np.loadtxt('data/test_pi.hmm')

def time_hmm(HMM_list, A, B, pi, obs_list):
	warnings.simplefilter("ignore") # ignore devide zero warning
	for obs in obs_list:
		for HMM in HMM_list:
			print '\nStarting "' + str(HMM)  + '", observation length: ' + str(len(obs))
			hmm = HMM(len(A), len(B[0]), A, B, pi)
			start = time.time()
			logprob, it = hmm.BaumWelch(obs, 1e-3, 10000)
			end = time.time()
#			hmm.printModel()
			print 'Returned after ' + str(it) + ' iterations.'
			print 'time: ' + str(end - start) + ' seconds.'
			print 'log P( O | lambda ) = ' + str(logprob)


# hmmList = [CHMM, FortranHMM, NumPyHMM, PyHMM, PySimpleHMM]
hmmList = [CHMM, FortranHMM]

time_hmm(hmmList, A, B, pi, obs_list)

