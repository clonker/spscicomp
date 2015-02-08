import numpy
import multiprocessing
import hmm.algorithms
import hmm.kernel.python


class Thread(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)

class Task(object):
    def __init__(self, A, B, pi, ob):
        self.ob = ob
        self.A  = A
        self.B  = B
        self.pi = pi 
    def __call__(self):
        return hmm.algorithms.noms_and_denoms(
            self.A, self.B, self.pi, self.ob, kernel=hmm.kernel.python)

def baum_welch_multiple(obs, A, B, pi, accuracy=1e-3, maxit=1000, kernel=hmm.kernel.python):
    K, N, M = len(obs), len(A), len(B[0])
    nomsA   = numpy.zeros((K,N,N), dtype=A.dtype)
    denomsA = numpy.zeros((K,N),   dtype=A.dtype)
    nomsB   = numpy.zeros((K,N,M), dtype=B.dtype)
    denomsB = numpy.zeros((K,N),   dtype=B.dtype)
    weights = numpy.zeros(K,       dtype=pi.dtype)
    gamma   = numpy.zeros((K,N),   dtype=A.dtype)

    # creating Thread pool here
    num_threads = multiprocessing.cpu_count()
    num_threads = min(num_threads, K)
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    pool = [ Thread(tasks, results) for i in xrange(num_threads) ]
    for thread in pool:
        thread.start()
        
    old_eps = 0.0
    it = 0
    new_eps = accuracy+1

    while (abs(new_eps - old_eps) > accuracy and it < maxit):
        for k in xrange(K):
            tasks.put(Task(A, B, pi, obs[k]))
        for k in xrange(K):
            weights[k], nomsA[k], denomsA[k], nomsB[k], denomsB[k], gamma[k] \
                = results.get()
            
        A, B, pi = hmm.algorithms.update_multiple(
            weights, nomsA, denomsA, nomsB, denomsB, gamma)

        if (it == 0):
            old_eps = 0
        else:
            old_eps = new_eps
        new_eps = numpy.sum(weights)
        it += 1
            
    # Finalize threading but shutting them down
    for k in xrange(K):
        tasks.put(None)
            
    for thread in pool:
        thread.join()

    return A, B, pi, new_eps, it
