import hmm.kernel.python
import numpy

def noms_and_denoms(A, B, pi, ob, kernel=hmm.kernel.python):
        T = len(ob)
        weight, alpha, scaling = kernel.forward(A, B, pi, ob)
        beta   = kernel.backward(A, B, pi, ob, scaling)
        nomA   = kernel.transition_counts(alpha, beta, A, B, ob)
        denomA = kernel.state_counts(alpha, beta, T-1)
        nomB   = kernel.symbol_counts(alpha, beta, ob, len(B[0]))
        gamma  = alpha[T-1]*beta[T-1]
        gamma /= numpy.sum(gamma)
        denomB = denomA + gamma
        return weight, nomA, denomA, nomB, denomB


def update_multiple(weights, noms_A, denoms_A, noms_B, denoms_B, dtype=numpy.float64):
    K, N, M = len(weights), len(noms_A[0]), len(noms_B[0,0])
    A = numpy.zeros((N,N), dtype=dtype)
    B = numpy.zeros((N,M), dtype=dtype)
    for i in range(N):
        nom_A_i   = numpy.zeros(N, dtype=dtype)
        denom_A_i = 0.0
        for k in range(K):
            nom_A_i   += weights[k] * noms_A[k,i,:]
            denom_A_i += weights[k] * denoms_A[k,i]
        A[i,:] = nom_A_i / denom_A_i
    for i in range(N):
        nom_B_i   = numpy.zeros(M, dtype=dtype)
        denom_B_i = 0.0
        for k in range(K):
            nom_B_i   += weights[k] * noms_B[k, i, :]
            denom_B_i += weights[k] * denoms_B[k, i]
        B[i,:] = nom_B_i / denom_B_i
    return A, B


def baum_welch_multiple(obs, A, B, pi, accuracy=1e-3, maxit=1000, kernel=hmm.kernel.python, dtype=numpy.float64):
    K, N, M = len(obs), len(A), len(B[0])
    nomsA   = numpy.zeros((K,N,N), dtype=dtype)
    denomsA = numpy.zeros((K,N),   dtype=dtype)
    nomsB   = numpy.zeros((K,N,M), dtype=dtype)
    denomsB = numpy.zeros((K,N),   dtype=dtype)
    weights = numpy.zeros((K),     dtype=dtype)

    old_probability = 0.0
    it      = 0
    new_probability = accuracy+1

    while (abs(new_probability - old_probability) > accuracy and it < maxit):
        for k in range(K):
            weights[k], nomsA[k], denomsA[k], nomsB[k], denomsB[k] = \
                    noms_and_denoms(A, B, pi, obs[k], kernel=kernel)
        
        A, B = update_multiple(weights, nomsA, denomsA, nomsB, denomsB)

        if (it == 0):
            old_probability = 0
        else:
            old_probability = new_probability        
        new_probability = numpy.sum(weights)
        it += 1

    return A, B, pi, new_probability, it

def update(gamma, xi, ob, M, dtype=numpy.float64):
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
    This function is part of the Bell-Welch algorithm for a single observation.

    See Also
    --------
    state_probabilities : to calculate `gamma`
    transition_probabilities : to calculate `xi`

    """
    T,N = len(ob), len(gamma[0])
    pi = numpy.zeros((N), dtype=dtype)
    A  = numpy.zeros((N,N), dtype=dtype)
    B  = numpy.zeros((N,M), dtype=dtype)
    for i in range(N):
        pi[i] = gamma[0,i]
    for i in range(N):
        gamma_sum = 0.0
        for t in range(T-1):
            gamma_sum += gamma[t,i]
        for j in range(N):
            A[i,j] = 0.0
            for t in range(T-1):
                A[i,j] += xi[t,i,j]
            A[i,j] /= gamma_sum
        gamma_sum += gamma[T-1, i]
        for k in range(M):
            B[i,k] = 0.0
            for t in range(T):
                if ob[t] == k:
                    B[i,k] += gamma[t,i]
            B[i,k] /= gamma_sum
    return (A, B, pi)

def baum_welch(ob, A, B, pi, accuracy=1e-3, maxit=1000, kernel=hmm.kernel.python, dtype=numpy.float32):
    """ Perform an optimization iteration with a given initial model.

    Locally maximize P(O|A,B,pi) in a neighborhood of (A,B,pi) by iterating
    `update`. Stops if the probability does not change or the maximal 
    iteration number is reached.

    Parameters
    ----------
    ob : numpy.array shape (T)
         observation sequence
    A : numpy.array shape (N,N)
        initial transition matrix
    B : numpy.array shape (N,M)
        initial symbol probabilities
    pi : numpy.array shape (N)
         initial distribution
    accuracy : float, optional
               ending criteria for the iteration
    maxit : int, optional
            ending criteria for the iteration
    kernel : module, optional
             module containing all functions to make calculations with
    dtype : { numpy.float32, numpy.float64 }, optional
            datatype to be used for the matrices.

    Returns
    -------
    A : numpy.array shape (N,N)
        new transition matrix
    B : numpy.array shape (N,M)
        new symbol probabilities
    pi : numpy.array shape (N)
         new initial distribution
    new_probability : dtype
                      log P( O | A,B,pi )
    it : int
         number of iterations done

    See Also
    --------
    kernel.python, kernel.c, kernel.fortran : possible kernels
    baum_welch_multiple : perform optimization with multiple observations.
    """
    it = 0
    T = len(ob)
    old_probability = 0.0
    new_probability, alpha, scaling = kernel.forward(A, B, pi, ob, dtype)
    while (abs(new_probability - old_probability) > accuracy and it < maxit):
        beta = kernel.backward(A, B, pi, ob, scaling, dtype)
        gamma = kernel.state_probabilities(alpha, beta, dtype)
        xi = kernel.transition_probabilities(alpha, beta, A, B, ob, dtype)
        A, B, pi = update(gamma, xi, ob, len(B[0]), dtype)
        old_probability = new_probability
        new_probability, alpha, scaling = kernel.forward(A, B, pi, ob, dtype)
        it += 1
    return (A, B, pi, new_probability, it)