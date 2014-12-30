from hmm import *
import matplotlib.pyplot as plt

A = np.loadtxt('models/bigModelA.dat')
B = np.loadtxt('models/bigModelB.dat')
pi = np.loadtxt('models/bigModelPi.dat')
hmmStart = HiddenMarkovModel(len(A), len(B[0]), A, B, pi)
print 'generating random sequence (with python...)'
obs = hmmStart.randomSequence(40000)

A = np.eye(10)
B = B.copy()
pi = pi.copy()
hmmTrain = HiddenMarkovModel(len(A), len(B[0]), A, B, pi)
# simBefore = similarity(hmmStart, hmmTrain, 1000000)
hmmTrain.BaumWelch(obs, 1e-3, 100, verbose=True)
# simAfter = similarity(hmmStart, hmmTrain, 1000000)

print 'model which generated the obs.'
hmmStart.printModel()
print 'trained model:'
hmmTrain.printModel()
#print 'Similarity Before Optimizing:', simBefore
#print 'Similarity After  Optimizing:', simAfter 

