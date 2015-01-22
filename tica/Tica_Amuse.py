__author__ = 'rickwg'

import numpy as np
import Tica_PrincipleComp as ticaPrinComp
import numpy.matlib as matlib
from common_data_importer import CommonBinaryFileDataImporter

class TicaAmuse:

    def __init__( self, i_inFileName = None, i_outFileName = "../testdata/tica_independentComp.npy", i_addEps = 1e-16, i_timeLag = 1 ):


        if  i_inFileName is not None:

            self.m_prinCompInst = ticaPrinComp.TicaPrinComp( i_inFileName = i_inFileName
                                                            ,i_outFileName = i_outFileName
                                                            ,i_addEpsilon = i_addEps
                                                            ,i_timeLag = i_timeLag)
            self.m_prinCompTL   = self.m_prinCompInst.getPrinCompTL(  )
            # self.m_prinCompTL      = ticaPrinComp.TicaPrinCompTimeLagged( '../testdata/covTestBinary.npy', i_outFileName )

    @profile
    def performAmuse( self, i_thresholdICs = 1 ):

        if np.isscalar( self.m_prinCompInst.param_timeLag ):

            self.m_prinCompInst.computeCovariance( )
            self.m_prinCompInst.computeEigenDecompCov( )
            self.m_prinCompTL.computeICs( i_thresholdICs )

         # todo: adaption for several time steps
