
from Tica_Amuse import TicaAmuse
from time import time

start = time()

amuse = TicaAmuse('../testdata/mixedSignalsBinary.npy', '../testdata/tica_sepSignals.npy', i_addEps = 1e-16)

# i_thresholdICs \in [0,1] is the amount of total variance to pick a suitable number of dimensions
amuse.performAmuse( i_thresholdICs = 0.5)

elapsed = time() - start
print(elapsed)
