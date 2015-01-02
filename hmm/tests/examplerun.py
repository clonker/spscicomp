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
from PyLessMemHMM import *
from ParallelHMM import *
from FortranParallelHMM import *
from CParallelHMM import *
from PyOpenCLHMM import *
from HybridCLHMM import *

print 'Read data...'

obs_list2 = [
	np.loadtxt('data/t1.100.dat', dtype='int'),
]

obs_list1 = [
	np.loadtxt('data/t1.2000.dat', dtype='int'),
]

obs_list3 = [
	np.loadtxt('data/t1.30000.dat', dtype='int'),
]

A = np.loadtxt('data/test_A.hmm', dtype='float64')
B = np.loadtxt('data/test_B.hmm', dtype='float64')
pi = np.loadtxt('data/test_pi.hmm', dtype='float64')

def time_hmm(HMM_list, A, B, pi, obs_list):
	warnings.simplefilter("ignore") # ignore devide zero warning
	for obs in obs_list:
		for HMM in HMM_list:
			print '\nStarting "' + str(HMM)  + '", observation length: ' + str(len(obs))
			hmm = HMM(len(A), len(B[0]), A, B, pi)
			start = time.time()
			logprob, it = hmm.BaumWelch(obs, 1e-3, 1)
			end = time.time()
			hmm.printModel()
			print 'Returned after ' + str(it) + ' iterations.'
			print 'time: ' + str(end - start) + ' seconds.'
			print 'log P( O | lambda ) = ' + str(logprob)


def time_multiple(HMM_list, A, B, pi, obs_list):
	for HMM in HMM_list:
		print '\nStarting "' + str(HMM)  + '", observations: ' + str(len(obs_list)) + 'x' + str(len(obs_list[0]))
#		for k in range(len(obs_list)):
#			print str(k+1) + '. list length: ' + str(len(obs_list[k]))
		hmm = HMM(len(A), len(B[0]), A, B, pi)
		start = time.time()
		logprob, it = hmm.BaumWelch_multiple(obs_list, 1e-3, 100)
		end = time.time()
		hmm.printModel()
		print 'Returned after ' + str(it) + ' iterations.'
		print 'time: ' + str(end - start) + ' seconds.'
		print 'log P( O | lambda ) = ' + str(logprob)

# hmmList = [CHMM, FortranHMM, NumPyHMM, PyHMM, PySimpleHMM, ParallelHMM]
# hmmList = [CHMM, FortranHMM]
# hmmList = [FortranHMM]
#hmmList2 = [PySimpleHMM, ParallelHMM, QueueHMM]
#hmmList1 = [FortranHMM, FortranParallelHMM, FortranQueueHMM, CHMM]
hmmList1 = [HybridCLHMM, CHMM32]
# time_multiple(hmmList2, A, B, pi, obs_list2*10)
#time_hmm(hmmList1, A, B, pi, obs_list3)
time_hmm(hmmList1, A, B, pi, obs_list1)
#time_multiple(hmmList1, A, B, pi, obs_list3*20)
