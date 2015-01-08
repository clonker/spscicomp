"""Python implementation of Hidden Markov Model kernel functions

This module is considered to be the reference for checking correctness of other
kernels. All implementations are being kept very simple, straight forward and
closely related to Rabiners [1] paper.

.. [1] Lawrence R. Rabiner, "A Tutorial on Hidden Markov Models and
   Selected Applications in Speech Recognition", Proceedings of the IEEE,
   vol. 77, issue 2
"""
import numpy as np

def forward_no_scaling(A, B, pi, ob, dtype=numpy.float64):
    """Compute P(ob|A,B,pi) and all forward coefficients. No scaling done.

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    ob : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    prob : floating number
           The probability to observe the sequence `ob` with the model given 
           by `A`, `B` and `pi`.
    alpha : np.array of floating numbers and shape (T,N)
            alpha[t,i] is the ith forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.

    See Also
    --------
    forward : Compute forward coefficients and scaling factors
    """
    T, N = len(ob), len(A)
    alpha = np.zeros((T,N), dtype=dtype)

    # initial values
    for i in range(N):
        alpha[0,i] = pi[i]*B[i,ob[0]]
    # induction
    for t in range(T-1):
        for j in range(N):
            alpha[t+1,j] = 0.0
            for i in range(N):
                alpha[t+1,j] += alpha[t,i] * A[i,j]
            alpha[t+1,j] *= B[j,ob[t+1]]
    prob = alpha[T-1].sum()
    return (prob, alpha)

def backward_no_scaling(A, B, pi, ob, dtype=numpy.float64):
    """Compute all backward coefficients. No scaling.

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    ob : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    beta : np.array of floating numbers and shape (T,N)
           beta[t,i] is the ith backward coefficient of time t

    See Also
    --------
    backward : Compute backward coefficients using given scaling factors.
    """
    T, N = len(ob), len(A)
    beta = np.zeros((T,N), dtype=dtype)

    # initital value
    for i in range(N):
        beta[T-1,i] = 1
    # induction
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t,i] = 0.0
            for j in range(N):
                beta[t,i] += A[i,j] * beta[t+1,j] * B[j,ob[t+1]]
    return beta

def forward(A, B, pi, ob, dtype=numpy.float64):
    """Compute P(ob|A,B,pi) and all forward coefficients. With scaling!

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    ob : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    prob : floating number
           The probability to observe the sequence `ob` with the model given 
           by `A`, `B` and `pi`.
    alpha : np.array of floating numbers and shape (T,N)
            alpha[t,i] is the ith forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.
    scaling : np.array of floating numbers and shape (T)
            scaling factors for each step in the calculation. can be used to
            rescale backward coefficients.

    See Also
    --------
    forward_no_scaling : Compute forward coefficients without scaling
    """
    T, N = len(ob), len(A)
    alpha = np.zeros((T,N), dtype=dtype)
    scale = np.zeros(T, dtype=dtype)
    
    # initial values
    for i in range(N):
        alpha[0,i] = pi[i] * B[i,ob[0]]
        scale[0] += alpha[0,i]
    for i in range(N):
        alpha[0,i] /= scale[0]

    # induction
    for t in range(T-1):
        for j in range(N):
            alpha[t+1,j] = 0.0
            for i in range(N):
                alpha[t+1,j] += alpha[t,i] * A[i,j]
            alpha[t+1,j] *= B[j, ob[t+1]]
            scale[t+1] += alpha[t+1,j]
        for j in range(N):
            alpha[t+1,j] /= scale[t+1]
        
    logprob = 0.0
    for t in range(T):
        logprob += np.log(scale[t])
    return (logprob, alpha, scale)


def backward(A, B, pi, ob, dtype=numpy.float64):
    """Compute all backward coefficients. With scaling!

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of floating numbers and shape (N)
         initial distribution
    ob : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    beta : np.array of floating numbers and shape (T,N)
            beta[t,i] is the ith forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.

    See Also
    --------
    backward_no_scaling : Compute backward coefficients without scaling
    """
    T, N = len(obs), self.N
    beta = np.zeros((T,N), dtype=dtype)
    for i in range(N):
        beta[T-1,i] = 1.0 / self.scale[T-1]
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t,i] = 0.0
            for j in range(N):
                beta[t,i] += self.A[i,j] * beta[t+1,j] * self.B[j,obs[t+1]] / self.scale[t]
    return beta


def gamma(alpha, beta, dtype=numpy.float64):
    """ Calculate the (T,N)-probabilty matrix for being in state i at time t.

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    dtype : item datatype [optional]

    Returns
    -------
    gamma : numpy.array shape (T,N)
            gamma[t,i] is the probabilty at time t to be in state i !

    Notes
    -----
    This function is independ of alpha and beta being scaled, as long as their
    scaling is independ in i.

    See Also
    --------
    forward, forward_no_scaling : to calculate `alpha`
    backward, backward_no_scaling : to calculate `beta`
    """
    gamma = alpha * beta;
    gamma = (gamma.T / gamma.sum(axis=0)).T # scaling for each row
    return gamma

def summed_gamma(alpha, beta, T_max=-1, dtype=numpy.float64):


def gamma_counts(ob, gamma, M, dtype=numpy.float64):

def transition_counts(alpha, beta, A, B, ob):


def summed_transition_counts(alpha, beta, A, B, ob):

def update(gamma, counts, ob):
