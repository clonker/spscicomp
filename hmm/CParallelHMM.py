from CHMM import *
from ParallelHMM import *

class CParallelHMM(CHMM, ParallelHMM):
	def __init__(self, N, M, A, B, pi):
		super(CParallelHMM, self).__init__(N, M, A, B, pi)
