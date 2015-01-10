import hmm.kernel.python
import numpy

def noms_and_denoms(A, B, pi, ob, kernel=hmm.kernel.python, dtype=numpy.float32):
    T = len(ob)
    weight, alpha, scaling = kernel.forward(A, B, pi, ob, dtype)
    beta   = kernel.backward(A, B, ob, scaling, dtype)
    gamma  = kernel.state_probabilities(alpha, beta, dtype)
    nomA   = kernel.transition_counts(alpha, beta, A, B, ob, dtype)
    denomA = kernel.state_counts(gamma, T-1, dtype)
    nomB   = kernel.symbol_counts(gamma, ob, len(B[0]), dtype)
    denomB = denomA + gamma[T-1]
    return weight, nomA, denomA, nomB, denomB


def update_multiple(weights, noms_A, denoms_A, noms_B, denoms_B, dtype=numpy.float32):
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


def baum_welch_multiple(obs, A, B, pi, accuracy=1e-3, maxit=1000, kernel=hmm.kernel.python, dtype=numpy.float32):
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
                    noms_and_denoms(A, B, pi, obs[k], kernel=kernel, dtype=dtype)
        
        A, B = update_multiple(weights, nomsA, denomsA, nomsB, denomsB, dtype=dtype)

        if (it == 0):
            old_probability = 0
        else:
            old_probability = new_probability
        new_probability = numpy.sum(weights)
        it += 1

    return A, B, pi, new_probability, it

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
    dtype : { numpy.float32, numpy.float32 }, optional
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
    T = len(ob)
    old_probability = 0.0
    it = 0
    new_probability = accuracy+1
    while (abs(new_probability - old_probability) > accuracy and it < maxit):
        probability, alpha, scaling = kernel.forward(A, B, pi, ob, dtype)

        print alpha

        beta = kernel.backward(A, B, ob, scaling, dtype)

        print beta

        gamma = kernel.state_probabilities(alpha, beta, dtype)
        xi = kernel.transition_probabilities(alpha, beta, A, B, ob, dtype)
        A, B, pi = kernel.update(gamma, xi, ob, len(B[0]), dtype)

        if it == 0:
            old_probability = 0
        else:
            old_probability = new_probability
        new_probability = probability
        it += 1
    return (A, B, pi, new_probability, it)
