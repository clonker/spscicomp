__author__ = 'rickwg'

from numpy import linalg as lina
import numpy as np
import numpy.matlib as matlib
import Tica_EigenDecomp as ticaEDecomp

class TicaPrinComp:
    """Naive class for computing principle components(PCs)."""

    def __init__(self, i_data):

        if None is not i_data and 0 < i_data.shape[0]:

            self.m_data             = np.asmatrix(i_data, dtype=np.float32)
            self.m_dataMeanFree     = np.array([])
            self.m_pc               = np.array([])
            self.m_pcNorm           = np.array([])
            self.m_covMat           = matlib.repmat(0.0, self.m_data.shape[1], self.m_data.shape[1])
            self.m_eigenDecomp      = ticaEDecomp.TicaEigenDecomp(None)
            self.m_additveEpsilon   = 1e-4

        else:

            self.m_data             = np.array([])
            self.m_dataMeanFree     = np.array([])
            self.m_pc               = np.array([])
            self.m_pcNorm           = np.array([])
            self.m_covMat           = np.array([])
            self.m_eigenDecomp      = ticaEDecomp.TicaEigenDecomp(None)
            self.m_additveEpsilon   = 1e-4


    # ---------------------------------------------------------------------------------------------#
    def makeDataMeanFree(self, i_data):

        o_dataMeanFree = i_data

        for i in range(i_data.shape[1]):

            o_dataMeanFree[:, i] = i_data[:, i] - np.mean(i_data[:, i], dtype=np.float32)

        return np.asmatrix(o_dataMeanFree, dtype=np.float32)

    #---------------------------------------------------------------------------------------------#
    def computeCovariance(self):

        if 0 < self.m_dataMeanFree.shape[0]:

            normFactor = 1 / (self.m_dataMeanFree.shape[0] - 1)

            # toDo: performant computation of covariance matrix
            #
            for i in range(self.m_dataMeanFree.shape[1]):

                for k in range(self.m_dataMeanFree.shape[1]):

                    col1 = self.m_dataMeanFree[:, i]
                    col2 = self.m_dataMeanFree[:, k]
                    self.m_covMat[i, k] = normFactor * ( col1.getT() * col2 )


    #---------------------------------------------------------------------------------------------#
    def computePC(self):

        self.m_dataMeanFree = self.makeDataMeanFree(self.m_data)
        self.computeCovariance()
        self.m_eigenDecomp.setMatrix(self.m_covMat)
        self.m_eigenDecomp.computeEigenDecomp()

        self.m_pc = self.m_dataMeanFree * self.m_eigenDecomp.m_eigenVecReal

    #---------------------------------------------------------------------------------------------#
    def normalizPC(self):

        normEigenVal = 1 / np.sqrt(self.m_eigenDecomp.m_eigenValReal+self.m_additveEpsilon)
        self.m_pcNorm = self.m_pc * np.diag(normEigenVal)



# ###################################################################################################

class TicaPrinCompTimeLagged(TicaPrinComp):
    """Naive class for computing principle components(PCs)."""

    def __init__(self, i_pcInstant, i_pcTimeLag):

        if None is not i_pcInstant and None is not i_pcTimeLag:

            super().__init__(None)

            if 0 < i_pcInstant.shape[0] and 0 < i_pcTimeLag.shape[0]:

                self.m_pcTimelag        = i_pcTimeLag
                self.m_pcInstant        = i_pcInstant
                self.m_covMatTimeLag    = matlib.repmat(0.0, self.m_pcTimelag.shape[1], self.m_pcTimelag.shape[1])
                self.m_ic               = matlib.repmat(0.0, self.m_pcTimelag.shape[0], self.m_pcTimelag.shape[1])
                self.m_covMatTimeLagSym = matlib.repmat(0.0, self.m_pcTimelag.shape[1], self.m_pcTimelag.shape[1])
                #self.m_eigenDecompTL    = ticaEDecomp.TicaEigenDecomp(None)

        else:

            self.m_dataTL           = np.array([])
            self.m_dataTLMeanFree   = np.array([])
            self.m_covMatTimeLag    = np.array([])
            self.m_pcTimelag        = np.array([])
            self.m_pcInstant        = np.array([])
            self.m_covMatTimeLagSym = np.array([])
            #self.m_eigenDecompTL    = ticaEDecomp.TicaEigenDecomp(None)


    #---------------------------------------------------------------------------------------------#
    def computeCovariance(self):

        normFactor = 1 / (self.m_pcTimelag.shape[0] - 1)

        # toDo: performant computation of covariance matrix
        #
        for i in range(self.m_pcTimelag.shape[1]):

            for k in range(self.m_pcTimelag.shape[1]):
                col1 = self.m_pcTimelag[:, i]
                col2 = self.m_pcInstant[:, k]
                self.m_covMatTimeLag[i, k] = normFactor * ( col1.getT() * col2 )

     #---------------------------------------------------------------------------------------------#
    def symmetrizeCovariance(self):

        self.m_covMatTimeLagSym = 0.5 * ( self.m_covMatTimeLag + self.m_covMatTimeLag.transpose() )

    #---------------------------------------------------------------------------------------------#
    def computePC(self):

        self.computeCovariance()
        self.symmetrizeCovariance()
        self.m_eigenDecomp.setMatrix(self.m_covMatTimeLagSym)
        self.m_eigenDecomp.computeEigenDecomp()

        self.m_ic = self.m_pcInstant * self.m_eigenDecomp.m_eigenVecReal