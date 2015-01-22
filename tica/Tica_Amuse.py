__author__ = 'rickwg'

import numpy as np
import Tica_PrincipleComp as ticaPrinComp
import numpy.matlib as matlib
from common_data_importer import CommonBinaryFileDataImporter

class TicaAmuse:

    def __init__( self, i_fileName = None, i_outFileName = "../testdata/tica_indComp.npy", i_addEps = 1e-16 ):


        if  i_fileName is not None:

            self.m_prinCompInst    = ticaPrinComp.TicaPrinComp( i_fileName = i_fileName, i_addEpsilon = i_addEps )
            self.m_prinCompTL      = ticaPrinComp.TicaPrinCompTimeLagged( self.m_prinCompInst.getOutFileName(), i_outFileName )

    def performAmuse( self, i_timeLag = 1, i_thresholdICs = 1 ):

        if np.isscalar( i_timeLag ):

            self.m_prinCompInst.computePC()
            self.m_prinCompInst.normalizPC()

            self.m_prinCompTL.computePC(i_thresholdICs)

         # todo: adaption for several time steps
