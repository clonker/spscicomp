import hmm.kernel.c
import numpy

def noms_and_denoms(A, B, pi, ob, kernel=hmm.kernel.c):
        T = len(ob)
        weight, alpha, scaling = kernel.forward(ob, A, B, pi)
        beta   = kernel.backward(ob, A, B, pi)
        nomA   = kernel.summed_counts(ob, alpha, beta, A, B)
        gamma  = kernel.gamma(alpha, beta)
        denomA = gamma[0:T-2].sum(axis=0)
        nomB   = kernel.gamma_counts(ob, gamma, M)
        denomB = denomA + gamma[T-1]
        return weight, nomA, denomA, nomB, denomB


def update_multiple(A, B, weights, noms_A, denoms_A, noms_B, denoms_B):
    K, N, M = len(weights), len(A), len(B[0])
    for i in range(N):
        nom_A_i   = numpy.zeros(N, dtype=A.dtype)
        denom_A_i = 0.0
        for k in range(K):
            nom_A_i   += weights[k] * noms_A[k,i,:]
            denom_A_i += weights[k] * denoms_A[k,i]
        A[i,:] = nom_A_i / denom_A_i
    for i in range(N):
        nom_B_i   = np.zeros(M, dtype=B.dtype)
        denom_B_i = 0.0
        for k in range(K):
            nom_B_i   += weights[k] * nomsB[k, i, :]
            denom_B_i += weights[k] * denomsB[k, i]
        B[i,:] = nom_B_i / denom_B_i
    return A, B


def baum_welch_multiple(obs, A, B, pi, accuracy=1e-3, maxit=1000, kernel=hmm.kernel.c):
    K, N, M = len(obs), len(A), len(B[0])
    nomsA   = numpy.zeros((K,N,N), dtype=A.dtype)
    denomsA = numpy.zeros((K,N),   dtype=A.dtype)
    nomsB   = numpy.zeros((K,N,M), dtype=B.dtype)
    denomsB = numpy.zeros((K,N),   dtype=B.dtype)
    weights = numpy.zeros((K),     dtype=pi.dtype) # lol...

    old_eps = 0.0
    it      = 0
    new_eps = accuracy+1

    while (abs(new_eps - old_eps) > accuracy and it < maxiter):
        for k in range(K):
            weights[k], nomsA[k], denomsA[k], nomsB[k], denomsB[k] = \
                    noms_and_denoms(A, B, pi, obs[k], kernel)
        
        A, B = update_multiple(weights, nomsA, denomsA, nomsB, denomsB)

        if (it == 0):
            old_eps = 0
        else:
            old_eps = new_eps        
            new_eps = np.sum(weights)
            it += 1

    return A, B, pi, new_eps, it

def baum_welch(ob, A, B, pi, accuracy=1e-3, maxit=1000, kernel=hmm.kernel.c):
    old_eps = 0.0
    it = 0
    T = len(ob)
    new_eps,alpha,scaling = kernel.forward(A, B, pi, obs)
    while (abs(new_eps - old_eps) > accuracy and it < maxit):
        beta = kernel.backward(A, B, pi, obs, scaling)
        gamma = kernel.computeGamma(alpha, beta)
        xi = kernel.counts(obs, alpha, beta)
        kernel.update(obs, gamma, xi)
        old_eps = new_eps
        new_eps, alpha = kernel.forward(obs)
        it += 1
    return (new_eps, it)