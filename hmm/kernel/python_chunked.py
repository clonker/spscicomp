"""Python implementation of Hidden Markov Model kernel functions

This module is considered to be the reference for checking correctness of other
kernels. All implementations are being kept very simple, straight forward and
closely related to Rabiners [1] paper.

.. [1] Lawrence R. Rabiner, "A Tutorial on Hidden Markov Models and
   Selected Applications in Speech Recognition", Proceedings of the IEEE,
   vol. 77, issue 2
"""
import numpy
from hmm.utility import ChunkedArray


def forward_no_scaling(A, B, pi, ob, dtype=numpy.float32):
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
    alpha : numpy.array of floating numbers and shape (T,N)
            alpha[t,i] is the i-th forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.

    See Also
    --------
    forward : Compute forward coefficients and scaling factors
    """
    T, N = len(ob), len(A)
    alpha = ChunkedArray(T, 10)

    # initial values
    alpha.set(0, numpy.zeros((N), dtype=dtype))
    for i in range(N):
        alpha.get(0)[i] = pi[i]*B[i, ob[0]]
    # induction
    for t in range(T-1):
        alpha.set(t, numpy.zeros((N), dtype=dtype))
        for j in range(N):
            #alpha[t+1, j] = 0.0
            for i in range(N):
                alpha.get(t+1)[j] += alpha.get(t)[i] * A[i, j]
            alpha.get(t+1)[j] *= B[j, ob[t+1]]
    prob = alpha.get(T-1).sum()
    return (prob, alpha)


def backward_no_scaling(A, B, ob, dtype=numpy.float32):
    """Compute all backward coefficients. No scaling.

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
    ob : numpy.array of integers and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    beta : np.array of floating numbers and shape (T,N)
           beta[t,i] is the i-th backward coefficient of time t

    See Also
    --------
    backward : Compute backward coefficients using given scaling factors.
    """
    T, N = len(ob), len(A)
    beta = ChunkedArray(T, 10)

    # initital value
    beta.set(T-1, numpy.zeros((N), dtype=dtype))
    for i in range(N):
        beta.get(T-1)[i] = 1
    # induction
    for t in range(T-2, -1, -1):
        for i in range(N):
            #beta.get(t)[i] = 0.0
            beta.set(t, numpy.zeros((N), dtype=dtype))
            for j in range(N):
                beta.get(t)[i] += A[i, j] * beta.get(t+1)[j] * B[j, ob[t+1]]
    return beta


def forward(A, B, pi, ob, dtype=numpy.float32):
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
            alpha[t,i] is the i-th forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.
    scaling : np.array of floating numbers and shape (T)
            scaling factors for each step in the calculation. can be used to
            rescale backward coefficients.

    See Also
    --------
    forward_no_scaling : Compute forward coefficients without scaling
    """
    T, N = len(ob), len(A)
    alpha = ChunkedArray(T, 10)  #  numpy.zeros((T,N), dtype=dtype)
    scale = ChunkedArray(T, 10)  #  numpy.zeros(T, dtype=dtype)
    # initial values
    alpha.set(0, numpy.zeros((N), dtype=dtype))
    for i in range(N):
        alpha.get(0)[i] = pi[i] * B[i, ob[0]]
        scale.set(0, scale.get(0) + alpha.get(0)[i])
    for i in range(N):
        alpha.get(0)[i] /= scale.get(0)

    # induction
    for t in range(T-1):
        for j in range(N):
            alpha.set(t+1, numpy.zeros((N), dtype=dtype)) #alpha[t+1,j] = 0.0
            for i in range(N):
                alpha.get(t+1)[j] += alpha.get(t)[i] * A[i, j]
            alpha(t+1)[j] *= B[j, ob[t+1]]
            scale.set(t+1, scale.get(t+1) + alpha.get(t+1)[j])
        for j in range(N):
            alpha(t+1)[j] /= scale.get(t+1)
    logprob = 0.0
    for t in range(T):
        logprob += numpy.log(scale.get(t))
    return (logprob, alpha, scale)


def backward(A, B, ob, scaling, dtype=numpy.float32):
    """Compute all backward coefficients. With scaling!

    Parameters
    ----------
    A : numpy.array of floating numbers and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of floating numbers and shape (N,M)
        symbol probability matrix for each hidden state
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
    T, N = len(ob), len(A)
    beta = ChunkedArray(T, 10)  #  numpy.zeros((T,N), dtype=dtype)
    beta.set(T-1, numpy.zeros((N), dtype=dtype))
    for i in range(N):
        beta.get(T-1)[i] = 1.0 / scaling.get(T-1)
    for t in range(T-2, -1, -1):
        beta.set(t, numpy.zeros((N), dtype=dtype))
        for i in range(N):
            #beta.get[t,i] = 0.0
            for j in range(N):
                beta.set(t)[i] += A[i, j] * beta.get(t+1)[j] * B[j, ob[t+1]] / scaling.get(t)
    return beta


def update(gamma, xi, ob, M, dtype=numpy.float32):
    """ Return an updated model for given state and transition counts.

    Parameters
    ----------
    gamma : numpy.array shape (T,N)
            state probabilities for each t 
    xi : numpy.array shape (T,N,N)
         transition probabilities for each t
    ob : numpy.array shape (T)
         observation sequence containing only symbols, i.e. ints in [0,M)

    Returns
    -------
    A : numpy.array (N,N)
        new transition matrix
    B : numpy.array (N,M)
        new symbol probabilities
    pi : numpy.array (N)
         new initial distribution
    dtype : { nupmy.float64, numpy.float32 }, optional

    Notes
    -----
    This function is part of the Baum-Welch algorithm for a single observation.

    See Also
    --------
    state_probabilities : to calculate `gamma`
    transition_probabilities : to calculate `xi`

    """
    T, N = len(ob), len(gamma[0])
    pi = numpy.zeros((N), dtype=dtype)
    A = numpy.zeros((N, N), dtype=dtype)
    B = numpy.zeros((N, M), dtype=dtype)
    for i in range(N):
        pi[i] = gamma.get(0)[i]
    for i in range(N):
        gamma_sum = 0.0
        for t in range(T-1):
            gamma_sum += gamma.get(t)[i]
        for j in range(N):
            A[i, j] = 0.0
            for t in range(T-1):
                A[i, j] += xi[t, i, j]
            A[i, j] /= gamma_sum
        gamma_sum += gamma.get(T-1)[i]
        for k in range(M):
            B[i, k] = 0.0
            for t in range(T):
                if ob[t] == k:
                    B[i, k] += gamma.get(t)[i]
            B[i, k] /= gamma_sum
    return (A, B, pi)


def state_probabilities(alpha, beta, dtype=numpy.float32):
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
    T, N = len(alpha), len(alpha[0])
    gamma = ChunkedArray(T, 10)  #  numpy.zeros((T, N), dtype=dtype)
    for t in range(T):
        gsum = 0.0
        gamma.set(t, numpy.zeros((N), dtype=dtype))
        for i in range(N):
            gamma.get(t)[i] = alpha.get(t)[i]*beta.get(t)[i]
            gsum += gamma.get(t)[i]
        for i in range(N):
            gamma.get(t)[i] /= gsum
    return gamma


def state_counts(gamma, T, dtype=numpy.float32):
    """ Sum the probabilities of being in state i to time t

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    dtype : item datatype [optional]

    Returns
    -------
    count : numpy.array shape (N)
            count[i] is the summed probabilty to be in state i !

    Notes
    -----
    This function is independ of alpha and beta being scaled, as long as their
    scaling is independ in i.

    See Also
    --------
    forward, forward_no_scaling : to calculate `alpha`
    backward, backward_no_scaling : to calculate `beta`
    """
    tsum = 0.0
    for t in range(T):
        tsum += gamma.get(t)[0]
    #return numpy.sum(gamma[0:T], axis=0)
    return tsum


def symbol_counts(gamma, ob, M, dtype=numpy.float32):
    """ Sum the observed probabilities to see symbol k in state i.

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    ob : numpy.array shape (T)
    M : integer
    dtype : item datatype, optional

    Returns
    -------
    counts : ...

    Notes
    -----
    This function is independ of alpha and beta being scaled, as long as their
    scaling is independ in i.

    See Also
    --------
    forward, forward_no_scaling : to calculate `alpha`
    backward, backward_no_scaling : to calculate `beta`
    """
    T, N = gamma.get_array_size(), len(gamma.get(0))
    counts = numpy.zeros((N, M), dtype=type)
    for t in range(T):
        for i in range(N):
            counts[i, ob[t]] += gamma.get(t)[i]
    return counts


def transition_probabilities(alpha, beta, A, B, ob, dtype=numpy.float32):
    """ Compute for each t the probability to transition from state i to state j.

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    A : numpy.array shape (N,N)
        transition matrix of the model
    B : numpy.array shape (N,M)
        symbol probabilty matrix of the model
    ob : numpy.array shape (T)
         observation sequence containing only symbols, i.e. ints in [0,M)
    dtype : item datatype [optional]

    Returns
    -------
    xi : numpy.array shape (T, N, N)
         xi[t, i, j] is the probability to transition from i to j at time t.

    Notes
    -----
    It does not matter if alpha or beta scaled or not, as long as there scaling
    does not depend on the second variable.

    See Also
    --------
    state_counts : calculate the probability to be in state i at time t
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`

    """
    T, N = len(ob), len(A)
    xi = ChunkedArray(T-1, 10)  # numpy.zeros((T-1, N, N), dtype=dtype)
    for t in range(T-1):
        xsum = 0.0
        xi.set(t, numpy.zeros((N, N), dtype=dtype))
        for i in range(N):
            for j in range(N):
                xi.get(t)[i, j] = alpha.get(t)[i] * A[i, j] * B[j, ob[t+1]] * beta.get(t+1)[j]
                xsum += xi.get(t)[i, j]
        for i in range(N):
            for j in range(N):
                xi.get(t)[i, j] /= xsum
    return xi


def transition_counts(alpha, beta, A, B, ob, dtype=numpy.float32):
    """ Sum for all t the probability to transition from state i to state j.

    Parameters
    ----------
    alpha : numpy.array shape (T,N)
            forward coefficients
    beta : numpy.array shape (T,N)
           backward coefficients
    A : numpy.array shape (N,N)
        transition matrix of the model
    B : numpy.array shape (N,M)
        symbol probabilty matrix of the model
    ob : numpy.array shape (T)
         observation sequence containing only symbols, i.e. ints in [0,M)
    dtype : item datatype [optional]

    Returns
    -------
    counts : numpy.array shape (N, N)
         counts[i, j] is the summed probability to transition from i to j 
         int time [0,T)

    Notes
    -----
    It does not matter if alpha or beta scaled or not, as long as there scaling
    does not depend on the second variable.

    See Also
    --------
    transition_probabilities : return the matrix of transition probabilities
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`
    """
    T, N = len(ob), len(A)
    xi = numpy.zeros((N, N), dtype=dtype)
    counts = numpy.zeros_like(xi)
    for t in range(T-1):
        xsum = 0.0
        for i in range(N):
            for j in range(N):
                xi[i, j] = alpha.get(t)[i] * A[i, j] * B[j, ob[t+1]] * beta.get(t+1)[j]
                xsum += xi[i, j]
        for i in range(N):
            for j in range(N):
                xi[i, j] /= xsum
        counts += xi
    return counts


def random_sequence(A, B, pi, T):
    """ Generate an observation sequence of length T from the model A, B, pi.

    Parameters
    ----------
    A : numpy.array shape (N,N)
        transition matrix of the model
    B : numpy.array shape (N,M)
        symbol probability matrix of the model
    pi : numpy.array shape (N)
         starting probability vector of the model

    Returns
    -------
    obs : numpy.array shape (T)
          observation sequence containing only symbols, i.e. ints in [0,M)

    Notes
    -----
    This function relies on the function draw_state(distr).

    See Also
    --------
    draw_state : draw the index of the state, obeying the probability
                 distribution vector distr

    """
    obs = numpy.zeros(T, dtype=numpy.int16)
    state = draw_state(pi)
    obs[0] = state
    for t in range(1, T):
        state = draw_state(A[state])
        obs[t] = draw_state(B[state])
    return obs


def draw_state(distr):
    x = numpy.random.random()
    D = len(distr)
    for state in range(D):
        if x < distr[state]:
            return state
        else:
            x -= distr[state]
    return state
