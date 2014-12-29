from hmm import *
import matplotlib.pyplot as plt

A = np.loadtxt('data/t1_A.hmm')
B = np.loadtxt('data/t1_B.hmm')
pi = np.loadtxt('data/t1_pi.hmm')

hmmStart = HiddenMarkovModel(len(A), len(B[0]), A, B, pi)

obs = hmmStart.randomSequence(33333)

for i in range(len(obs)):
	print obs[i]
