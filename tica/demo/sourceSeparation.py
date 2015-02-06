from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np

from Tica_Amuse import TicaAmuse
from common_data_importer import CommonBinaryFileDataImporter
from time import time

mixingImporter  = CommonBinaryFileDataImporter('../testdata/mixedSignalsBinary.npy')
origImporter    = CommonBinaryFileDataImporter('../testdata/origSignalsBinary.npy')

mixingData      = mixingImporter.get_data(len(mixingImporter._file))
origData        = origImporter.get_data(len(origImporter._file))


start = time()

amuse = TicaAmuse('../testdata/mixedSignalsBinary.npy', '../testdata/tica_sepSignals.npy', i_addEps = 1e-16, i_timeLag=1)
amuse.performAmuse( i_numDomComp = 2 )

elapsed = time() - start
print(elapsed)


sepImporter  = CommonBinaryFileDataImporter('../testdata/tica_sepSignals.npy')
sepData      = sepImporter.get_data(len(sepImporter._file))

x = np.arange(0.0, np.pi, np.pi/sepData.shape[0])

plt.subplot(2, 1, 1)
line1, = plt.plot(x, origData[:, 0], label = '$\sin(10x)$')
line2, = plt.plot(x, origData[:, 1], label = '$\cos(50x)$')
plt.title('Original Signals')
plt.ylim(-1.5, 1.5)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

plt.subplot(2, 1, 2)
plt.plot(x, mixingData[:, 0])
plt.plot(x, mixingData[:, 1])
plt.title('Mixed Signals')
plt.ylim(-1.5, 1.5)

plt.show()


plt.subplot(2, 1, 1)
plt.plot(x, sepData[:, 0])
plt.title('Separated Signals')
plt.subplot(2, 1, 2)
plt.plot(x, sepData[:, 1])
plt.show()
