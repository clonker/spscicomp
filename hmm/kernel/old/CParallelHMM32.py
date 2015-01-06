from CHMM32 import *
from ParallelHMM import *

class CParallelHMM32(CHMM32, ParallelHMM):
	def __init__(self, N, M, A, B, pi):
		PySimpleHMM.__init__(self, N, M, A, B, pi, np.float32)
