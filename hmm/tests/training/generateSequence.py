from hmm import *
import matplotlib.pyplot as plt

A = np.loadtxt('models/startmodelA_1.dat')
B = np.loadtxt('models/startmodelB_1.dat').T
pi = np.loadtxt('models/startmodelPi_1.dat')

hmmStart = HiddenMarkovModel(len(A), len(B), A, B, pi)

obs = hmmStart.randomSequence(33333)

for i in range(len(obs)):
	print obs[i]
