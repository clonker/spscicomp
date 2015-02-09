#!/usr/bin/python
import spscicomp.hmm.utility
import sys

A, B, pi = spscicomp.hmm.utility.get_models()[sys.argv[1]]
seq = spscicomp.hmm.utility.random_sequence(A, B, pi, int(sys.argv[2]))
for ob in seq:
	print ob
