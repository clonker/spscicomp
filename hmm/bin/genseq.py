#!/usr/bin/python
import numpy as np
import hmm.utility

<<<<<<< HEAD
A = np.array(	[[0.3, 0.7, 0.0],
				 [0.1, 0.2, 0.7],
				 [0.0, 0.4, 0.6]])

B = np.array(	[[0.3, 0.3, 0.3, 0.1],
				 [0.1, 0.1, 0.1, 0.7],
				 [0.2, 0.2, 0.2, 0.4]])

pi = np.array(	[[0.2],
				 [0.4],
				 [0.4]])

obs = hmm.utility.random_sequence(A, B, pi, 1000, kernel=hmm.kernel.python)

print obs
=======
import hmm.utility
import hmm.models
import sys

A, B, pi = hmm.models.get_models()[sys.argv[1]]

seq = hmm.utility.generate_sequence(A, B, pi, int(sys.argv[2]))

for ob in seq:
	print ob
>>>>>>> 05f862519fba04175e35aff7b0e9b0b690dfd395
