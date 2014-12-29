from hmm import *
import matplotlib.pyplot as plt

A = np.loadtxt('models/startmodelA_1.dat')
B = np.loadtxt('models/startmodelB_1.dat')
pi = np.loadtxt('models/startmodelPi_1.dat')

hmmStart = HiddenMarkovModel(len(A), len(B[0]), A, B, pi)

print 'Generating observation sequence. Using Model'
hmmStart.printModel()
obs = hmmStart.randomSequence(36632)

A = np.array([
	[0.5, 0.2, 0.3],
	[0.2, 0.5, 0.3],
	[0.2, 0.3, 0.5]
])
B = np.array([
	[0.3, 0.7],
	[0.5, 0.5],
	[0.6, 0.4]
])
pi = pi.copy()

hmmTrain = HiddenMarkovModel(len(A), len(B[0]), A, B, pi)
print 'Starting model training. Initial model:'
hmmTrain.printModel()

# simBefore = similarity(hmmStart, hmmTrain, 1000000)
likeli = hmmTrain.BaumWelch(obs,1e-3, 20000, verbose=True)
#simAfter = similarity(hmmStart, hmmTrain, 1000000)

print 'trained model:'
hmmTrain.printModel()
#print 'Similarity Before Optimizing:', simBefore
#print 'Similarity After  Optimizing:', simAfter

plt.plot(likeli, color='black')
plt.show()
