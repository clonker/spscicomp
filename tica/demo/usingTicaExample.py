
from Tica_Amuse import TicaAmuse
from time import time

start = time()

amuse = TicaAmuse('../testdata/mixedSignalsBinary.npy', '../testdata/tica_sepSignals.npy', i_addEps = 1e-8)

# i_numDomComp number of needed components/dimensions
amuse.performAmuse( i_numDomComp = 2 )

elapsed = time() - start
print(elapsed)
