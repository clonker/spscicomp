import hmm.lib.c as ext
import numpy as np


#@profile
def forward(A, B, pi, ob):
    ob = np.array(ob, dtype=np.int32)
    if (A.dtype == np.float32):
        return ext.forward32(A, B, pi, ob)
    else:
        return ext.forward(A, B, pi, ob)        

