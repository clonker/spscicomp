#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
# import hmm.py to have access to Hidden Markov Model kernel
from hmm import *

obs = np.loadtxt('data/t1.33333.dat', dtype='int')

A = np.loadtxt('data/test_A.hmm')
B = np.loadtxt('data/test_B.hmm')
pi = np.loadtxt('data/test_pi.hmm')

hmm = HiddenMarkovModel(len(A), len(B[0]), A, B, pi)
hmm.BaumWelch(obs, 1e-3, 1000)
hmm.printModel()

