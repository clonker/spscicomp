from FortranHMM import *
from ParallelHMM import *

class FortranParallelHMM(FortranHMM, ParallelHMM):
	def __init__(self, N, M, A, B, pi):
		super(FortranParallelHMM, self).__init__(N, M, A, B, pi)
