#!/usr/bin/python

from PySimpleHMM import *
import sys

A = np.loadtxt('data/t1_A.hmm')
B = np.loadtxt('data/t1_B.hmm')
pi = np.loadtxt('data/t1_pi.hmm')

hmm = PySimpleHMM(len(A), len(B[0]), A, B, pi)
obs = hmm.randomSequence(int(sys.argv[1]))

for o in obs:
	print o
