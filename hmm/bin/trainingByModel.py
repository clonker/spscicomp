import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
import hmm.algorithms
import hmm.utility
import hmm.kernel.c

models = hmm.utility.get_models()

A, B, pi = models['t2']
print 'A\n', A
print 'B\n', B
print 'pi\n', pi

obs = np.loadtxt(
	"/home/chris/git/spscicomp/hmm/data/t2.1000.dat",
	dtype=np.int16)

print obs

A, B, pi = hmm.utility.generate_startmodel(len(A),len(B[0]),dtype=np.float32)

A, B, pi, prob, it = hmm.algorithms.baum_welch(
    obs,
    A,
    B,
    pi,
    accuracy=-1.,
    maxit=50,
    kernel=hmm.kernel.python,
    dtype=np.float32)

print A 
print B
print pi
print it
