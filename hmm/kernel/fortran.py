import hmm.lib.fortran as ext
import numpy

def forward(A, B, pi, ob, dtype=numpy.float32):
    return ext.forward(A, B, pi, ob)

def backward(A, B, ob, scaling, dtype=numpy.float32):    
    return ext.backward(A, B, ob, scaling)

def state_probabilities(alpha, beta, dtype=numpy.float32):
    return ext.computegamma(alpha, beta)

def state_counts(gamma, T, dtype=numpy.float32):
    return ext.computedenoma(gamma)

def update(gamma, xi, ob, M, dtype=numpy.float32):
    return ext.update(ob, gamma, xi, M)

def symbol_counts(gamma, ob, M, dtype=numpy.float32):
    return ext.computenomb(ob, gamma, M)

def transition_probabilities(alpha, beta, A, B, ob, dtype=numpy.float32):
    return ext.computexi(A, B, ob, alpha, beta)

def transition_counts(alpha, beta, A, B, ob, dtype=numpy.float32):
    return ext.computenoma(A, B, ob, alpha, beta)
