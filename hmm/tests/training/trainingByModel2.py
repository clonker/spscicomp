from hmm import *
import matplotlib.pyplot as plt

A = np.loadtxt('models/bigModelA.dat')
B = np.loadtxt('models/bigModelB.dat').T
pi = np.loadtxt('models/bigModelPi.dat')
print A
print B
print pi
hmmStart = HiddenMarkovModel(len(A), len(B), A, B, pi)

obs = hmmStart.randomSequence(2000000)

A = np.eye(10)
B = B.copy()
pi = pi.copy()

hmmTrain = HiddenMarkovModel(len(A), len(B), A, B, pi)
simBefore = similarity(hmmStart, hmmTrain, 1000000)
A,B,pi,likeli = hmmTrain.optimize(obs,0.5, 500, verbose=True)
hmmTrain.printModel()
simAfter = similarity(hmmStart, hmmTrain, 1000000)

print 'model which generated the obs.'
hmmStart.printModel()
print 'trained model:'
hmmTrain.printModel()
print 'Similarity Before Optimizing:', simBefore
print 'Similarity After  Optimizing:', simAfter 

plt.plot(likeli, color='black')
plt.show()
