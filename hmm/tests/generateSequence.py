from PySimpleHMM import *

A = np.loadtxt('data/t1_A.hmm')
B = np.loadtxt('data/t1_B.hmm')
pi = np.loadtxt('data/t1_pi.hmm')

hmm = PySimpleHMM(len(A), len(B[0]), A, B, pi)
obs = hmm.randomSequence(100000)

for o in obs:
	print o
