import numpy as np
import sys

def random_by_dist(distribution):
	x = np.random.random();
	for n in range(len(distribution)):
		if x < (distribution[n]):
			return n;
		else:
			x -= distribution[n];
	#print 'Reached exceptional state in random_by_dist()'
	return n

def compare(model1, model2, obsLength):
	"""Quantify the similarity of two models, based on an observation 
	sequence of model2."""
	observation = model2.randomSequence(obsLength)
	prob1, _ = model1.forward(observation)
	prob2, _ = model2.forward(observation)
	result = prob2 - prob1
	return result / float(obsLength)

def similarity(model1, model2, obsLength):
	"""Give a symmetric realisation of the similarity.

	Average the unsymmetric similarities compare(model1, model2)
	and compare(model2, model1).

	"""
	com1 = compare(model1, model2, obsLength)
	com2 = compare(model2, model1, obsLength)
	return 0.5*(com1 + com2)

# def model_from_file(model, filename):
#
