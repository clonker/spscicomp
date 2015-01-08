import hmm.lib.c as ext
import numpy

def forward_no_scaling(A, B, pi, ob, dtype=numpy.float32):
    if A.dtype != dtype or B.dtype != dtype or pi.dtype != dtype:
            raise ValueError
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.forward_no_scaling32(A, B, pi, ob)
    if dtype == numpy.float64:
        return ext.forward_no_scaling(A, B, pi, ob)
    else:
        raise ValueError

def forward(A, B, pi, ob, dtype=numpy.float32):
    if A.dtype != dtype or B.dtype != dtype or pi.dtype != dtype:
            raise ValueError
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.forward32(A, B, pi, ob)
    if dtype == numpy.float64:
        return ext.forward(A, B, pi, ob)
    else:
        raise ValueError

def backward_no_scaling(A, B, ob, dtype=numpy.float32):
    if A.dtype != dtype or B.dtype != dtype:
            raise ValueError
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.backward_no_scaling32(A, B, ob)
    if dtype == numpy.float64:
        return ext.backward_no_scaling(A, B, ob)
    else:
        raise ValueError

def backward(A, B, ob, scaling, dtype=numpy.float32):
    if A.dtype != dtype or B.dtype != dtype or scaling.dtype != dtype:
        raise ValueError
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.backward32(A, B, ob, scaling)
    if dtype == numpy.float64:
        return ext.backward(A, B, ob, scaling)
    else:
        raise ValueError

def state_probabilities(alpha, beta, dtype=numpy.float32):
    if beta.dtype != dtype or alpha.dtype != dtype:
            raise ValueError
    if dtype == numpy.float32:
        return ext.state_probabilities32(alpha, beta)
    if dtype == numpy.float64:
        return ext.state_probabilities(alpha, beta)
    else:
        raise ValueError

def state_counts(gamma, T, dtype=numpy.float32):
    if gamma.dtype != dtype:
            raise ValueError
    if dtype == numpy.float32:
        return ext.state_counts32(gamma, T)
    if dtype == numpy.float64:
        return ext.state_counts(gamma, T)
    else:
        raise ValueError

def symbol_counts(gamma, ob, M, dtype=numpy.float32):
    if gamma.dtype != dtype:
            raise ValueError
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.symbol_counts32(gamma, ob, M)
    if dtype == numpy.float64:
        return ext.symbol_counts(gamma, ob, M)
    else:
        raise ValueError

def transition_probabilities(alpha, beta, A, B, ob, dtype=numpy.float32):
    if A.dtype != dtype or B.dtype != dtype or alpha.dtype != dtype or beta.dtype != dtype:
            raise ValueError
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.transition_probabilities32(alpha, beta, A, B, ob)
    if dtype == numpy.float64:
        return ext.transition_probabilities(alpha, beta, A, B, ob)
    else:
        raise ValueError

def transition_counts(alpha, beta, A, B, ob, dtype=numpy.float32):
    if A.dtype != dtype or B.dtype != dtype or alpha.dtype != dtype or beta.dtype != dtype:
            raise ValueError
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.transition_counts32(alpha, beta, A, B, ob)
    if dtype == numpy.float64:
        return ext.transition_counts(alpha, beta, A, B, ob)
    else:
        raise ValueError

