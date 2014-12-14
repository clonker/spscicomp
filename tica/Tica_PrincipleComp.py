__author__ = 'rickwg'

from numpy import linalg as lina
import numpy as np
import numpy.matlib as matlib
import Tica_EigenDecomp as ticaEDecomp

class TicaPrinComp:
    """Naive class for computing principle components(PCs)."""

    def __init__(self, i_data, i_addEpsilon = 1e-16):

        if  i_data is not None and 0 < i_data.shape[0]:

            self.m_data             = np.asmatrix(i_data, dtype=np.float32)
            self.m_dataMeanFree     = np.array([])
            self.m_pc               = np.array([])
            self.m_pcNorm           = np.array([])
            self.m_covMat           = matlib.repmat(0.0, self.m_data.shape[1], self.m_data.shape[1])
            self.m_eigenDecomp      = ticaEDecomp.TicaEigenDecomp(None)
            self.m_addEpsilon       = i_addEpsilon
            
        else:

            self.m_data             = np.array([])
            self.m_dataMeanFree     = np.array([])
            self.m_pc               = np.array([])
            self.m_pcNorm           = np.array([])
            self.m_covMat           = np.array([])
            self.m_eigenDecomp      = ticaEDecomp.TicaEigenDecomp(None)
            self.m_addEpsilon       = i_addEpsilon


    # ---------------------------------------------------------------------------------------------#
    def makeDataMeanFree(self, i_data):

        o_dataMeanFree = matlib.repmat(0.0, i_data.shape[0], i_data.shape[1])

        o_dataMeanFree = i_data - np.mean(i_data, dtype=np.float32, axis=0)

        return o_dataMeanFree

    #---------------------------------------------------------------------------------------------#
    def computeCovariance(self):

        if 0 < self.m_dataMeanFree.shape[0]:

            normFactor = 1.0 / (self.m_dataMeanFree.shape[0] - 1)
            self.m_covMat = self.m_dataMeanFree.T * self.m_dataMeanFree
            self.m_covMat = normFactor * self.m_covMat

    #---------------------------------------------------------------------------------------------#
    def computePC(self, i_amountOfTotalVariance = 1.0):

        self.m_dataMeanFree = self.makeDataMeanFree(self.m_data)
        self.computeCovariance()
        self.m_eigenDecomp.computeEigenDecomp(self.m_covMat)

        dc = self.m_eigenDecomp.m_eigenVecReal.shape[1]
        if 1 > i_amountOfTotalVariance:

            dc = self.calcNumbOfDomComps(i_amountOfTotalVariance)

        self.m_pc = self.m_dataMeanFree * self.m_eigenDecomp.m_eigenVecReal[:, 0:dc]

    #---------------------------------------------------------------------------------------------#
    def normalizPC(self):

        lamb = 1.0 / np.sqrt(self.m_eigenDecomp.m_eigenValReal+self.m_addEpsilon)
        self.m_pcNorm = self.m_pc * np.diag(lamb)

    #---------------------------------------------------------------------------------------------#
    def getNormalizedPCs(self):

        if self.m_pcNorm is not None and 0 < self.m_pcNorm.shape[0]:

            return self.m_pcNorm

        else:

            return print("There are no normalized Priciple Components")

    #---------------------------------------------------------------------------------------------#
    def getPCs(self):

        return self.m_pc

    #---------------------------------------------------------------------------------------------#
    def calcNumbOfDomComps(self, i_amountOfTotalVariance):
        
        totalVariance = np.sum(self.m_eigenDecomp.m_eigenValReal)
        sumEigVal = 0
        for i, e in enumerate(self.m_eigenDecomp.m_eigenValReal):

            sumEigVal += e
            if sumEigVal/totalVariance >= i_amountOfTotalVariance:

                return i+1



####################################################################################################

class TicaPrinCompTimeLagged(TicaPrinComp):
    """Naive class for computing principle components(PCs)."""

    def __init__(self, i_data, i_timeLag = 1):

        super().__init__(i_data = i_data)
        if i_data is not None:

            if 0 < i_data.shape[0]:

                self.m_timeLag          = i_timeLag
                self.m_covMatTimeLag    = matlib.repmat(0.0, self.m_data.shape[1], self.m_data.shape[1])
                self.m_pcTimeLag        = matlib.repmat(0.0, self.m_data.shape[1], self.m_data.shape[1])
                self.m_covMatTimeLagSym = matlib.repmat(0.0, self.m_data.shape[1], self.m_data.shape[1])

        else:

            self.m_timeLag          = i_timeLag
            self.m_covMatTimeLag    = np.array([])
            self.m_pcTimeLag        = np.array([])
            self.m_covMatTimeLagSym = np.array([])

    #---------------------------------------------------------------------------------------------#
    def computeCovariance(self):

        if 0 < self.m_data.shape[0]:

            m = self.m_data.shape[0]
            normFactor = 1.0 / (m - self.m_timeLag - 1)
            self.m_covMatTimeLag = self.m_data[0:m-self.m_timeLag, :].T * self.m_data[self.m_timeLag:m, :]
            self.m_covMatTimeLag *= normFactor

    #---------------------------------------------------------------------------------------------#
    def symmetrizeCovariance(self):

        self.m_covMatTimeLagSym = 0.5 * ( self.m_covMatTimeLag + self.m_covMatTimeLag.transpose() )

    #---------------------------------------------------------------------------------------------#
    def setTimeLag(self, i_timeLag):

        self.m_timeLag = i_timeLag

    #---------------------------------------------------------------------------------------------#
    def computePC(self, i_amountOfTotalVariance = 1):

        # self.m_data = self.makeDataMeanFree(self.m_data)
        self.computeCovariance()
        self.symmetrizeCovariance()
        self.m_eigenDecomp.computeEigenDecomp(self.m_covMatTimeLagSym)

        dc = self.m_eigenDecomp.m_eigenVecReal.shape[1]
        if 1 > i_amountOfTotalVariance:

            dc = self.calcNumbOfDomComps(i_amountOfTotalVariance)

        self.m_pcTimeLag = self.m_data * self.m_eigenDecomp.m_eigenVecReal[:, 0:dc]

    #---------------------------------------------------------------------------------------------#
    def getPCsTimeLag(self):

        return self.m_pcTimeLag

    #---------------------------------------------------------------------------------------------#
    def setData(self, i_data):

        self.m_data = i_data





                

