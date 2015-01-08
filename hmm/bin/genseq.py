#!/usr/bin/python
import hmm.utility
import hmm.models
import sys
A, B, pi = hmm.models.get_models()[sys.argv[1]]
seq = hmm.utility.random_sequence(A, B, pi, int(sys.argv[2]))
for ob in seq:
	print ob
