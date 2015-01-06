import numpy as np
import sys
import warnings
import time

def random_by_dist(distribution):
    x = np.random.random();
    for n in range(len(distribution)):
        if x < (distribution[n]):
            return n;
        else:
            x -= distribution[n];
    #print 'Reached exceptional state in random_by_dist()'
    return n

def compare(model1, model2, obsLength):
    """Quantify the similarity of two models, based on an observation 
    sequence of model2."""
    observation = model2.randomSequence(obsLength)
    prob1, _ = model1.forward(observation)
    prob2, _ = model2.forward(observation)
    result = prob2 - prob1
    return result / float(obsLength)

def similarity(model1, model2, obsLength):
    """Give a symmetric realisation of the similarity.

    Average the unsymmetric similarities compare(model1, model2)
    and compare(model2, model1).

    """
    com1 = compare(model1, model2, obsLength)
    com2 = compare(model2, model1, obsLength)
    return 0.5*(com1 + com2)

def time_baum_welch(HMM_list, A, B, pi, obs_list):
    warnings.simplefilter("ignore") # ignore devide zero warning
    for obs in obs_list:
        for HMM in HMM_list:
            print '\nStarting "' + str(HMM)  + '", observation length: ' + str(len(obs))
            hmm = HMM(len(A), len(B[0]), A, B, pi)
            start = time.time()
            logprob, it = hmm.BaumWelch(obs, 0, 10)
            end = time.time()
            hmm.printModel()
            print 'Returned after ' + str(it) + ' iterations.'
            print 'time: ' + str(end - start) + ' seconds.'
            print 'log P( O | lambda ) = ' + str(logprob)


def time_baum_welch_multiple(HMM_list, A, B, pi, obs_list, eps=1e-3, maxit=1000):
    warnings.simplefilter("ignore") # ignore devide zero warning
    for HMM in HMM_list:
        print '\nStarting "' + str(HMM)  + '", observations: ' + str(len(obs_list)) + 'x' + str(len(obs_list[0]))
        hmm = HMM(len(A), len(B[0]), A, B, pi)
        start = time.time()
        logprob, it = hmm.BaumWelch_multiple(obs_list, eps, maxit)
        end = time.time()
        hmm.printModel()
        print 'Returned after ' + str(it) + ' iterations.'
        print 'time: ' + str(end - start) + ' seconds.'
        print 'log P( O | lambda ) = ' + str(logprob)

# def model_from_file(model, filename):
#
