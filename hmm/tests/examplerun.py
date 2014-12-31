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

print 'Read data...'

obs_list2 = [
#	np.loadtxt('data/t2.5.dat', dtype='int'),
#	np.loadtxt('data/t1.100.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.1.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.2.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.3.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.4.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.5.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.6.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.7.dat', dtype='int'),
#	np.loadtxt('data/t1.2000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.1000000.dat', dtype='int'),
#	np.loadtxt('data/t1.3000000.dat', dtype='int'),
#	np.loadtxt('data/t1.5000000.dat', dtype='int')
]

obs_list1 = [
#	np.loadtxt('data/t2.5.dat', dtype='int'),
#	np.loadtxt('data/t1.100.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.1.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.2.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.3.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.4.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.5.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.6.dat', dtype='int'),
#	np.loadtxt('data/t1.1000.7.dat', dtype='int'),
#	np.loadtxt('data/t1.2000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
#	np.loadtxt('data/t1.100000.dat', dtype='int'),
	np.loadtxt('data/t1.1000000.dat', dtype='int'),
#	np.loadtxt('data/t1.3000000.dat', dtype='int'),
#	np.loadtxt('data/t1.5000000.dat', dtype='int')
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
			logprob, it = hmm.BaumWelch(obs, 1e-3, 100)
			end = time.time()
			hmm.printModel()
			print 'Returned after ' + str(it) + ' iterations.'
			print 'time: ' + str(end - start) + ' seconds.'
			print 'log P( O | lambda ) = ' + str(logprob)


# hmmList = [CHMM, FortranHMM, NumPyHMM, PyHMM, PySimpleHMM, ParallelHMM]
# hmmList = [CHMM, FortranHMM]
# hmmList = [FortranHMM]
# hmmList = [PySimpleHMM, ParallelHMM]
hmmList = [FortranHMM, FortranParallelHMM]

time_hmm(hmmList, A, B, pi, obs_list1)

def time_multiple(HMM_list, A, B, pi, obs_list):
	for HMM in HMM_list:
		print '\nStarting "' + str(HMM)  + '", observations: ' + str(len(obs_list)) + ' each length 100000.'
		hmm = HMM(len(A), len(B[0]), A, B, pi)
		start = time.time()
		logprob, it = hmm.BaumWelch_multiple(obs_list, 1e-3, 100)
		end = time.time()
		hmm.printModel()
		print 'Returned after ' + str(it) + ' iterations.'
		print 'time: ' + str(end - start) + ' seconds.'
		print 'log P( O | lambda ) = ' + str(logprob)


time_multiple(hmmList, A, B, pi, obs_list2)
