from hmm_ext import forward
import numpy as np

A = np.array([[2./3., 1./3.],[1./3., 2./3.]])
B = np.array([[1., 0.],[0.,1.]])
pi = np.array([1./3.,1./3.])
obs = np.array([0, 1, 1])

T = len(obs)
N = len(A)

alpha = np.zeros((T,N), dtype='double')
scale = np.zeros(T, dtype='double')

print alpha,scale

forward(alpha, scale, A, B, pi, obs)

print alpha,scale
