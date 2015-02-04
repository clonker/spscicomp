import numpy as np
import Tica_PrincipleComp as ticaPrinComp
import numpy.matlib as matlib
from common_data_importer import CommonBinaryFileDataImporter

class TicaAmuse:
    """
    Implementation of AMUSE-Algorithm, which basically uses functionality of :class:`.TicaPrinComp`.
    A :class:`.CommonDataImporter` object to be used for importing the numerical data.

    :param i_inFileName: A filename of the binary file which loads the data
    :type i_inFileName: string

    :param i_outFileName: A filename of the binary file which stored the results.
    :type i_outFileName: string

    :param i_addEps: A damping parameter to avoid dividing by zero in the normalization part of the amuse algorithmen.
    :type i_addEps: float

    :param i_timeLag: In this setting the data has time-dependencies where i_timeLag is some lag constant.
    :type i_timeLag: int
    """

    def __init__( self, i_inFileName = None, i_outFileName = "../testdata/tica_independentComp.npy", i_addEps = 1e-16, i_timeLag = 1 ):

        if i_inFileName is not None:

            self.m_prinCompInst = ticaPrinComp.TicaPrinComp( i_inFileName = i_inFileName
                                                            ,i_outFileName = i_outFileName
                                                            ,i_addEpsilon = i_addEps
                                                            ,i_timeLag = i_timeLag)
            self.m_prinCompTL   = self.m_prinCompInst.getPrinCompTL(  )
            # self.m_prinCompTL      = ticaPrinComp.TicaPrinCompTimeLagged( '../testdata/covTestBinary.npy', i_outFileName )

    #@profile
    def performAmuse( self, i_numDomComp = 1 ):
        """
        Runs the AMUSE-Algorithm and stores the results in a binary file with the stated filename(see above `i_outFileName`).

        :param i_numDomComp: Number of independent components which are needed.
        :type i_numDomComp: int
        """

        if np.isscalar( self.m_prinCompInst.param_timeLag ):

            self.m_prinCompInst.computeCovariance( )
            self.m_prinCompInst.computeEigenDecompCov( )
            self.m_prinCompTL.computeICs( i_numDomComp )