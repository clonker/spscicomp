__author__ = 'rickwg'

from numpy import linalg as lina
import numpy as np
import numpy.matlib as matlib

class TicaEigenDecomp:

    def __init__(self, i_matrix):

        if None == i_matrix:

            self.m_matrix        = []
            self.m_eigenVec      = []
            self.m_eigenVal      = []
            self.m_eigenVecReal  = []
            self.m_eigenValReal  = []

        elif 0 < i_matrix.shape[0] and 0 < i_matrix.shape[1]:

            self.m_matrix        = np.asmatrix(i_matrix, dtype=np.float32)
            self.m_eigenVec      = matlib.repmat(0.0, self.m_matrix.shape[0], self.m_matrix.shape[1])
            self.m_eigenVal      = matlib.repmat(0.0, self.m_matrix.shape[0], 1)
            self.m_eigenVecReal  = matlib.repmat(0.0, self.m_matrix.shape[0], self.m_matrix.shape[1])
            self.m_eigenValReal  = matlib.repmat(0.0, self.m_matrix.shape[0], 1)

        else:

            print("Check matrix dimension!")

    #---------------------------------------------------------------------------------------------#
    def setMatrix(self, i_matrix):

        self.m_matrix = np.asmatrix(i_matrix, dtype=np.float32)

    #---------------------------------------------------------------------------------------------#
    def reorderingEigenV(self):
        """ Reordering of eigenvalues and corresponding eigenvectors by the largest eigenvalue.
            Ordering only for real eigenvalues."""

        reOrdIndex          = np.argsort(self.m_eigenVal)
        self.m_eigenValReal = self.m_eigenVal[reOrdIndex[::-1]]
        self.m_eigenVecReal = self.m_eigenVec[:, reOrdIndex[::-1]]

    #---------------------------------------------------------------------------------------------#
    def computeEigenDecomp(self):

        self.m_eigenVal, self.m_eigenVec = lina.eig(self.m_matrix)

        if not any( 1e-15 < self.m_eigenVal.imag ):

            self.reorderingEigenV()