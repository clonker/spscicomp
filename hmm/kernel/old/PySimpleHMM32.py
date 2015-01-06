import numpy as np
from PySimpleHMM import *


class PySimpleHMM32(PySimpleHMM):
    def __init__(self, N, M, A, B, pi):
        PySimpleHMM.__init__(self, N, M, A, B, pi, np.float32)
