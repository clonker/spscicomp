from hmm import *
import matplotlib.pyplot as plt

A = np.loadtxt('data/startmodelA_1.dat')
B = np.loadtxt('data/startmodelB_1.dat').T
pi = np.loadtxt('data/startmodelPi_1.dat')


hmm = HiddenMarkovModel(len(A), len(B), A, B, pi)
obs = hmm.randomSequence(1000000)
A,B,pi,likeli = hmm.optimize(obs, 50)

print np.round(A,3)
print np.round(B,3)
print np.round(pi,3)

plt.plot(likeli, color='black')
plt.show()
