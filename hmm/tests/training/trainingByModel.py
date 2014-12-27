from hmm import *
import matplotlib.pyplot as plt

A = np.loadtxt('models/startmodelA_1.dat')
B = np.loadtxt('models/startmodelB_1.dat').T
pi = np.loadtxt('models/startmodelPi_1.dat')

hmmStart = HiddenMarkovModel(len(A), len(B), A, B, pi)


print 'Generating observation sequence. Using Model'
hmmStart.printModel()
obs = hmmStart.randomSequence(200000)

A = np.array([
	[0.5, 0.2, 0.3],
	[0.2, 0.5, 0.3],
	[0.2, 0.3, 0.5]
])
B = np.array([
	[0.3, 0.5, 0.6],
	[0.7, 0.5, 0.4]
])
pi = pi.copy()

hmmTrain = HiddenMarkovModel(len(A), len(B), A, B, pi)
print 'Starting model training. Initial model:'
hmmTrain.printModel()

# simBefore = similarity(hmmStart, hmmTrain, 1000000)
A,B,pi,likeli = hmmTrain.optimize(obs,1e-4, 20000, verbose=True)
# simAfter = similarity(hmmStart, hmmTrain, 1000000)

print 'trained model:'
hmmTrain.printModel()
#print 'Similarity Before Optimizing:', simBefore
#print 'Similarity After  Optimizing:', simAfter 

#plt.plot(likeli, color='black')
#plt.show()
