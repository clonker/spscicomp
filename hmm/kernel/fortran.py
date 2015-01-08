import hmm.lib.fortran as ext
import numpy

def forward_no_scaling(A, B, pi, ob, dtype=numpy.float64):
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.forward_no_scaling32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.forward_no_scaling(A, B, pi, ob)
    else
        raise ValueError

def forward(A, B, pi, ob, dtype=numpy.float32):
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.forward32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.forward(A, B, pi, ob)
    else
        raise ValueError

def backward_no_scaling(A, B, pi, scaling, ob, dtype=numpy.float64):
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.backward_no_scaling32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.backward_no_scaling(A, B, pi, ob)
    else
        raise ValueError

def backward(A, B, pi, scaling, ob, dtype=numpy.float64):
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.backward32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.backward(A, B, pi, ob)
    else
        raise ValueError

def gamma(alpha, beta, dtype=numpy.float32):
    if dtype == numpy.float32:
        return ext.gamma32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.gamma(A, B, pi, ob)
    else
        raise ValueError

def summed_gamma(alpha, beta, T, dtype=numpy.float64):
    if dtype == numpy.float32:
        return ext.summed_gamma32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.summed_gamma(A, B, pi, ob)
    else
        raise ValueError

def counts(alpha, beta, A, B, ob):
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.counts32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.counts(A, B, pi, ob)
    else
        raise ValueError

def summed_counts(alpha, beta, A, B, ob):
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.summed_counts32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.summed_counts(A, B, pi, ob)
    else
        raise ValueError

def update(gamma, counts, ob):
    if ob.dtype != numpy.int16:
        ob = numpy.array(ob, dtype=numpy.int16)
    if dtype == numpy.float32:
        return ext.update32(A, B, pi, ob)
    else if dtype == numpy.float64:
        return ext.update(A, B, pi, ob)
    else
        raise ValueError