#!/usr/bin/python
import numpy
import hmm.algorithms
import hmm.concurrent
import hmm.utility
import hmm.kernel.python
import hmm.kernel.c


def hmm_baum_welch(A, B, pi, observation, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32):
    """ update the model (A, B, pi) with the observation

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    observation : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B
    maxit : do forward and backward maxit times. Default is 1000
    kernel : kernel to use. default is hmm.kernel.c
    accuracy : if difference of log probability is smaller than accuracy, then
               abort the operation. Default is -1, so iteration will not aborted
    dtype : dtype of calculations. Default is numpy.float32

    Returns
    -------
    new model (A, B, pi) optimized by observation
    """
    A, B, pi, prob, it = hmm.algorithms.baum_welch(obs, A, B, pi, maxit=maxit, kernel=kernel, accuracy=accuracy, dtype=dtype)
    return A, B, pi


def hmm_baum_welch_file(A, B, pi, observation_file, observation_length, observation_count, observation_dtype=numpy.int16, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32):
    """ updates the model (A, B, pi) with observation_count observations stored in observation_file
    different orders in observationsequences result in different model

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    observation_file : file with a binary representation of a numpy array of integers and shape (T) 
                       observation sequence of integer between 0 and M, used as indices in B
    observation_length : length of one observationsequence
    observation_count : number of in observation_file stored observationsequences
    observation_dtype : dtype of observations. Default is numpy.int16
    maxit : do forward and backward maxit times. Default is 1000
    kernel : kernel to use. default is hmm.kernel.c
    accuracy : if difference of log probability is smaller than accuracy, then
               abort the operation. Default is -1, so iteration will not aborted
    dtype : dtype of calculations. Default is numpy.float32

    Returns
    -------
    new model (A, B, pi) optimized by observations in observation_file
    """
    A1 = A
    B1 = B
    pi1 = pi
    for i in range(observation_count):
        obs = hmm.utility.get_observation_part(observation_file, observation_length, observation_count, dtype=observation_dtype)
        A1, B1, pi1 = hmm_baum_welch(A1, B1, pi1, obs, maxit=maxit, kernel=kernel, accuracy=accuracy, dtype=dtype)
    return A1, B1, pi1


def hmm_baum_welch_multiple(A, B, pi, observations, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32):
    """ update the model (A, B, pi) with the observationslist
    all optimisation are done with the original model

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    observation : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B
    maxit : do forward and backward maxit times. Default is 1000
    kernel : kernel to use. default is hmm.kernel.c
    accuracy : if difference of log probability is smaller than accuracy, then
               abort the operation. Default is -1, so iteration will not aborted
    dtype : dtype of calculations. Default is numpy.float32


    Returns
    -------
    new model (A, B, pi) optimized by observations in observation_file
    """
    A1, B1, pi1, prob, it = \
        hmm.algorithms.baum_welch_multiple(
            observations, A, B, pi, 
            maxit=maxit, kernel=kernel, accuracy=accuracy, dtype=numpy.float32
        )
    return A1, B1, pi1


def hmm_baum_welch_multiple_file(A, B, pi, observation_file, observation_length, observation_count, observation_dtype=numpy.int16, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32):
    """ updates the model (A, B, pi) with observation_count observations stored in observation_file
    different orders in observationsequences result in different model

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    observation_file : file with a binary representation of a numpy array of integers and shape (T) 
                       observation sequence of integer between 0 and M, used as indices in B
    observation_length : length of one observationsequence
    observation_count : number of in observation_file stored observationsequences
    observation_dtype : dtype of observation. Default is numpy.int16
    maxit : do forward and backward maxit times. Default is 1000
    kernel : kernel to use. default is hmm.kernel.c
    accuracy : if difference of log probability is smaller than accuracy, then
               abort the operation. Default is -1, so iteration will not aborted
    dtype : dtype of calculations. Default is numpy.float32

    Returns
    -------
    new model (A, B, pi) optimized by observations in observation_file
    """
    K, N, M = observation_count, len(A), len(B[0])
    nomsA   = numpy.zeros((K,N,N), dtype=dtype)
    denomsA = numpy.zeros((K,N),   dtype=dtype)
    nomsB   = numpy.zeros((K,N,M), dtype=dtype)
    denomsB = numpy.zeros((K,N),   dtype=dtype)
    weights = numpy.zeros((K),     dtype=dtype)
    gamma_0 = numpy.zeros((K,N),   dtype=dtype)

    old_probability = 0.0
    it      = 0
    new_probability = accuracy+1

    while (abs(new_probability - old_probability) > accuracy and it < maxit):
        for k in range(K):
            observation = hmm.utility.get_observation_part(observation_file, observation_length, k, observation_dtype)
            weights[k], nomsA[k], denomsA[k], nomsB[k], denomsB[k], gamma_0[k] = \
                    hmm.algorithms.noms_and_denoms(A, B, pi, observation, kernel=kernel, dtype=dtype)
        
        A, B, pi = hmm.algorithms.update_multiple(weights, nomsA, denomsA, nomsB, denomsB, gamma_0, dtype=dtype)

        if (it == 0):
            old_probability = 0
        else:
            old_probability = new_probability
        new_probability = numpy.sum(weights)
        it += 1

    return A, B, pi


#obs = numpy.loadtxt('data/hmm1.10000.dat', dtype=numpy.int16)
#A, B, pi = hmm.utility.get_models()['t2']
#A, B, pi = hmm_baum_welch(A, B, pi, obs, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32)

#A, B, pi = hmm.utility.get_models()['t2']
#A, B, pi = hmm_baum_welch_file(A, B, pi, 'data/hmm1.10000.bin', 100, 100, observation_dtype=numpy.int16, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32)

#A, B, pi = hmm.utility.get_models()['t2']
#obfile = numpy.loadtxt('data/hmm1.10000.dat', dtype=numpy.int16)
#observations = [obfile[x:x+100] for x in xrange(0, len(obfile), 100)]
#A, B, pi = hmm_baum_welch_multiple(A, B, pi, observations, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32)

#A, B, pi = hmm.utility.get_models()['t2']
#hmm_baum_welch_multiple_file(A, B, pi, 'data/hmm1.10000.bin', 100, 100, observation_dtype=numpy.int16, maxit=1000, kernel=hmm.kernel.c, accuracy=-1, dtype=numpy.float32)

#print A
#print B
#print pi
