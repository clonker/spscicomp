from numpy import linalg as lina
import numpy as np
import numpy.matlib as matlib

class TicaEigenDecomp:
    """
    A class that post processes the results of the eigen decomposition of :class:`.numpy.linalg.eig`.
    Performs a reordering of the eigen values and the corresponding eigen vectors.
    Here are considered only real eigen values.

    :param i_matrix: A matrix from which the eigen decomposition will realized.
    :type i_matrix: numpy.array[]
    """

    def __init__(self, i_matrix):

        if i_matrix is None:

            self.m_matrix        = np.array([])
            self.m_eigenVec      = np.array([])
            self.m_eigenVal      = np.array([])
            self.m_eigenVecReal  = np.array([])
            self.m_eigenValReal  = np.array([])

        elif 0 < i_matrix.shape[0] and 0 < i_matrix.shape[1]:

            self.m_matrix        = np.asmatrix(i_matrix, dtype=np.float32)
            self.m_eigenVec      = matlib.repmat(0.0, self.m_matrix.shape[0], self.m_matrix.shape[1])
            self.m_eigenVal      = matlib.repmat(0.0, self.m_matrix.shape[0], 1)
            self.m_eigenVecReal  = matlib.repmat(0.0, self.m_matrix.shape[0], self.m_matrix.shape[1])
            self.m_eigenValReal  = matlib.repmat(0.0, self.m_matrix.shape[0], 1)

        else:

            print("Check matrix dimension!")

    #---------------------------------------------------------------------------------------------#
    def reorderingEigenV(self):
        """ 
        Reordering of eigenvalues and corresponding eigenvectors by the largest eigenvalue.
        Ordering only for real eigenvalues.
        """

        reOrdIndex          = np.argsort(self.m_eigenVal)
        self.m_eigenValReal = self.m_eigenVal[reOrdIndex[::-1]]
        self.m_eigenVecReal = self.m_eigenVec[:, reOrdIndex[::-1]]

    #---------------------------------------------------------------------------------------------#
    def computeEigenDecomp(self, i_matrix):
        """
        Performs the eigen decomposition of :class:`.numpy.linalg` and rearrange(see :func:`.reorderingEigenV`)
        the eigenvalues if they are real.
        If the eigenvalues have a imaginary part a warning will logged, the imaginary parts will be set to zero
        and the real parts will be rearranged.
        """

        self.m_matrix = np.asmatrix(i_matrix, dtype=np.float32)
        self.m_eigenVal, self.m_eigenVec = lina.eig(self.m_matrix)

        if not any( 1e-15 < self.m_eigenVal.imag ):

            self.reorderingEigenV()

        else:

            print('TICA: Warning: eigenvalues of covariance matrix have imaginary part!')
            im = np.asarray(self.m_eigenVal.imag)
            im.fill(0.0)
            self.m_eigenVal.imag = im;
            self.reorderingEigenV()