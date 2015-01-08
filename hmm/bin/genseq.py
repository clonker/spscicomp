#!/usr/bin/python
import numpy as np
import hmm.utility

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
