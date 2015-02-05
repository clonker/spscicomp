import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
import hmm.algorithms
import hmm.utility
import hmm.kernel.c

models = hmm.utility.get_models()

A, B, pi = models['hmm_1']
print 'A\n', A
print 'B\n', B
print 'pi\n', pi

ob = np.loadtxt(
	"/home/chris/git/spscicomp/hmm/data/hmm1.100000.dat",
	dtype=np.int16)
d = len(ob)/10
obs = [ ob[ x*d : x*d + d -1] for x in range(10)]

#A, B, pi = hmm.utility.generate_startmodel(len(A),len(B[0]),dtype=np.float32)
A, B, pi = models['t3']

A1, B1, pi1, prob, it = hmm.algorithms.baum_welch(
    ob,
    A,
    B,
    pi,
    accuracy=-1.,
    maxit=500,
    kernel=hmm.kernel.c,
    dtype=np.float32)
print "results baum_welch"
print A1
print B1
print pi1
print it

#A, B, pi = hmm.utility.generate_startmodel(len(A),len(B[0]),dtype=np.float32)
A, B, pi = models['t3']

A1, B1, pi1, prob, it = hmm.algorithms.baum_welch_multiple(
	obs,
	A,
	B,
	pi,
	accuracy=-1.,
	maxit = 500,
	kernel=hmm.kernel.c,
	dtype=np.float32)
print "results multiple"
print A1
print B1
print pi1
print it
