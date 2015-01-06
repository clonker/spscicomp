import numpy as np
from PySimpleHMM import *
import multiprocessing

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
    def __init__(self, hmm, ob):
        self.hmm = hmm
        self.ob = ob
    def __call__(self):
        return self.hmm.process_obs(self.ob)
    def __str__(self):
        return 'observation length %d' % len(self.ob)

class ParallelHMM(PySimpleHMM):
    def __init__(self, N, M, A, B, pi):
        super(ParallelHMM, self).__init__(N, M, A, B, pi)

    @profile
    def BaumWelch_multiple(self, obss, accuracy, maxiter):
        K, N, M = len(obss), self.N, self.M
        nomsA = np.zeros((K,N,N), dtype=self.dtype)
        denomsA = np.zeros((K,N), dtype=self.dtype)
        nomsB = np.zeros((K,N,M), dtype=self.dtype)
        denomsB = np.zeros((K,N), dtype=self.dtype)
        weights = np.zeros(K, dtype=self.dtype)

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

        while (abs(new_eps - old_eps) > accuracy and it < maxiter):
            
            for k in xrange(K):
                tasks.put(Task(self, obss[k]))
            for k in xrange(K):
                weights[k], nomsA[k], denomsA[k], nomsB[k], denomsB[k] = results.get()
            
            self.update_multiple(weights, nomsA, denomsA, nomsB, denomsB)

            if (it == 0):
                old_eps = 0
            else:
                old_eps = new_eps
            new_eps = np.sum(weights)
            it += 1
            
#        tasks.join()
        for k in range(K):
            tasks.put(None)
            
        for thread in pool:
            thread.join()

        return new_eps, it
