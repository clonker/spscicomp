import numpy as np
import numpy.matlib as matlib
import Tica_EigenDecomp as ticaEDecomp
from common_data_importer import CommonBinaryFileDataImporter
from array import array
import os

try:
    import ticaC

    use_extension = True
    print('TICA: Using C extension')
except:
    use_extension = False
    print('TICA: C extension not found, using Python implementation')

# short patch
use_extension = False

class TicaPrinComp:
    """
    Implementation of computation of principle components highly adapted for the amuse algorithm.
    The class :class:`.TicaPrinComp` contains the subclass :class:`.TicaPrinCompTimeLagged`.
    A :class:`.CommonDataImporter` object to be used for importing the numerical data.

    :param i_inFileName: A filename of the binary file which loads the data
    :type i_inFileName: string

    :param i_outFileName: A filename of the binary file in which the results are stored
    :type i_outFileName: string

    :param i_addEpsilon: A damping parameter to avoid dividing by zero in the normalization part of the amuse algorithm.
    :type i_addEps: float

    :param i_timeLag: In this setting the data has time-dependencies where i_timeLag is some lag constant.
    :type i_timeLag: int
    """

    def __init__(self, i_inFileName = None, i_outFileName = "../testdata/tica_tempOutput.npy", i_addEpsilon = 1e-16,
                 i_timeLag = 1):

        self.param_fileSizeThreshold = 500 * 1e+6  # file size in byte
        self.param_outFileName = i_outFileName

        if i_inFileName is not None:

            self.m_fileSize = os.path.getsize(i_inFileName)
            self.m_dataImporter = CommonBinaryFileDataImporter(i_inFileName)
            # self.m_dataImporter.create_out_file(self.param_outFileName)

            self.param_addEpsilon = i_addEpsilon
            self.param_chunkSize = None
            self.param_timeLag = i_timeLag
            self.computeChunkSize()
            print("Chunk Size")
            print(self.param_chunkSize)
            self.m_covMat = np.array([])
            self.m_eigenDecomp = ticaEDecomp.TicaEigenDecomp(None)
            self.m_colMeans = None

        else:

            self.m_eigenDecomp = ticaEDecomp.TicaEigenDecomp(None)

    # ---------------------------------------------------------------------------------------------#

    def getPrinCompTL(self):

        return self.TicaPrinCompTimeLagged(self)

    # ---------------------------------------------------------------------------------------------#
    def getTempFileName(self):

        return self.param_outFileName

    # ---------------------------------------------------------------------------------------------#
    def computeChunkSize(self):
        """
        This function computes in a naive way the chunk size, which is used to load a data chunk from the memory map
        of :class:`CommonDataImporter`.
        If the file size of the input file is greater than the threshold a bisection
        of the number of rows of the input file will be performed.
        """

        if self.m_fileSize < self.param_fileSizeThreshold:

            self.param_chunkSize = self.m_dataImporter.get_shape_inFile()[0]

        else:

            fileSize = self.m_fileSize

            if fileSize % 2 is not 0:

                fileSize += 1

            fileSize /= 2

            dimData = self.m_dataImporter.get_shape_inFile()[0]
            if dimData % 2 is not 0:

                self.param_chunkSize = ( dimData + 1 ) / 2

            else:

                self.param_chunkSize = dimData / 2

            while fileSize > self.param_fileSizeThreshold:

                self.param_chunkSize /= 2
                fileSize /= 2

    # ---------------------------------------------------------------------------------------------#
    def computeColMeans(self):
        """
        Computes mean per column of the input data.
        """

        self.m_dataImporter.rewind()

        dataChunk = np.asarray(self.m_dataImporter.get_data(self.param_chunkSize), dtype=np.float32)
        shapeData = dataChunk.shape
        dimData = shapeData[0]
        self.m_colMeans = np.sum(dataChunk[:, 0:shapeData[1]], dtype = np.float32, axis = 0)
        del dataChunk

        while self.m_dataImporter.has_more_data():

            dataChunk = np.asarray(self.m_dataImporter.get_data(self.param_chunkSize), dtype=np.float32)
            dimData += len(dataChunk)
            self.m_colMeans += np.sum(dataChunk[:, 0:shapeData[1]], dtype = np.float32, axis = 0)
            del dataChunk

        self.m_colMeans *= 1.0 / dimData


    # ---------------------------------------------------------------------------------------------#
    def makeDataMeanFree(self):

        self.computeColMeans()
        self.m_dataImporter.rewind()

        dataChunk = self.m_dataImporter.get_data(self.param_chunkSize)
        shapeData = dataChunk.shape
        meanFreeChunk = dataChunk[:, 0:shapeData[1]] - self.m_colMeans
        self.m_dataImporter.write_data(meanFreeChunk)

        while self.m_dataImporter.has_more_data():

            dataChunk = self.m_dataImporter.get_data(self.param_chunkSize)
            meanFreeChunk = dataChunk[:, 0:shapeData[1]] - self.m_colMeans
            self.m_dataImporter.write_data(meanFreeChunk)

    # ----------------------------------------------------------------------------------------------#
    def computeCovariance(self):
        """
        This function computes chunk by chunk the instantaneous covariance matrix :math:`C`.
        If a c-extension is available, then it will used by the flag `use_extension`
        """

        self.computeColMeans()
        self.m_dataImporter.rewind()

        shapeData = self.m_dataImporter.get_shape_inFile()
        #use_extension = False
        if use_extension is True:
            self.m_covMat = ticaC.computeCov(self.m_dataImporter.get_data,
                                             self.m_dataImporter.has_more_data,
                                             self.m_colMeans,
                                             self.param_chunkSize,
                                             shapeData[0])
        else:
            dataChunk = np.asarray(self.m_dataImporter.get_data(self.param_chunkSize)[:, :], dtype=np.float32)
            #dataChunk = self.m_dataImporter.get_data(self.param_chunkSize)
            #dataChunk[:, 0:shapeData[1]] -= self.m_colMeans
            meanFreeChunk = dataChunk[:, 0:shapeData[1]] - self.m_colMeans
            del dataChunk
            self.m_covMat = np.dot(meanFreeChunk.transpose(), meanFreeChunk)
            del meanFreeChunk

            while self.m_dataImporter.has_more_data():

                dataChunk = np.asarray(self.m_dataImporter.get_data(self.param_chunkSize)[:, :], dtype=np.float32)
                meanFreeChunk = dataChunk[:, 0:shapeData[1]] - self.m_colMeans
                del dataChunk
                self.m_covMat += np.dot(meanFreeChunk.transpose(), meanFreeChunk)
                del meanFreeChunk

            self.m_covMat *= 1.0 / (shapeData[0] - 1.0)

    # ---------------------------------------------------------------------------------------------#
    def computePCs(self, i_dataChunk, i_domComp):
        """
        This function computes the principle components of the input data :math:`X`.
        Let :math:`\\Gamma` be the matrix which contains the ordered eigenvectors of the instantaneous
        covariance matrix :math:`C`. The principle components :math:`Y` are computed
        by :math:`Y = \\Gamma^{T}(X-mean(X))`.

        :param i_dataChunk: A data chunk as a subset of hole data.
        :type i_dataChunk: numpy.array
        :param i_domComp: Number of dominated components which are considered.
        :type i_domComp: int
        :return o_pc: The principle components :math:`Y`.
        :rtype: numpy.array
        """

        if not 1 < i_dataChunk.shape[0]:

            meanFreeChunk = i_dataChunk[:, 0:i_dataChunk.shape[1]] - self.m_colMeans
            o_pc = np.dot(matlib.asmatrix(meanFreeChunk), self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp])
            del meanFreeChunk

        else:

            meanFreeChunk = i_dataChunk[:, 0:i_dataChunk.shape[1]] - self.m_colMeans
            o_pc = np.dot( meanFreeChunk, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp])
            del meanFreeChunk

        return o_pc

    # ---------------------------------------------------------------------------------------------#
    def computePC(self, i_amountOfTotalVariance = 1.0):

        self.makeDataMeanFree()
        self.computeCovariance()
        self.m_eigenDecomp.computeEigenDecomp(self.m_covMat)

        domComp = self.m_eigenDecomp.m_eigenVecReal.shape[1]
        if 1 > i_amountOfTotalVariance:

            domComp = self.calcNumbOfDomComps(i_amountOfTotalVariance)

        self.performTransformation(domComp)

    # ---------------------------------------------------------------------------------------------#
    def computeEigenDecompCov(self):

        self.m_eigenDecomp.computeEigenDecomp(self.m_covMat)

    # ---------------------------------------------------------------------------------------------#
    def normalizePCs(self, i_pcsChunk):
        """
        This function computes the normalizes the principle components :math:`Y` of the input data :math:`X`.
        Let :math:`\\Lambda` be a diagonal matrix with the ordered eigenvalues of the instantaneous
        #covariance matrix :math:`C`. The normalized principle components :math:`\\tilde{Y}` are computed
        by :math:`\\tilde{Y} = \\Lambda^{-\\frac{1}{2}}Y)`.

        :param i_pcsChunk: A chunk of principle components.
        :type i_pcsChunk: numpy.array

        :return o_pcNorm: The normalized principle components :math:`\\tilde{Y}`.
        :rtype: numpy.array
        """

        lamb = 1.0 / np.sqrt(self.m_eigenDecomp.m_eigenValReal + self.param_addEpsilon)

        if not 1 < i_pcsChunk.shape[0]:

            o_pcNorm = np.dot(matlib.asmatrix(i_pcsChunk), np.diag(lamb))

        else:

            o_pcNorm = np.dot(i_pcsChunk, np.diag(lamb))

        return o_pcNorm

    # ---------------------------------------------------------------------------------------------#
    def calcNumbOfDomComps(self, i_amountOfTotalVariance):
        """
        `Not used actually!`
        This function returns, by a given amount of the total variance, the number of relevant
        principle components.

        :param i_amountOfTotalVariance: Amount of total variance.
        :type i_amountOfTotalVariance: float

        :return : Return the number of dominant principle components.
        :rtype: int
        """

        totalVariance = np.sum(self.m_eigenDecomp.m_eigenValReal)
        sumEigVal = 0
        for i, e in enumerate(self.m_eigenDecomp.m_eigenValReal):

            sumEigVal += e
            if sumEigVal / totalVariance >= i_amountOfTotalVariance:

                return i + 1

    # ---------------------------------------------------------------------------------------------#
    def getOutFileName(self):

        return self.param_outFileName

    # ##################################################################################################

    class TicaPrinCompTimeLagged:
        """
        A subclass contained in :class:`.TicaPrinComp`. This class contains the time-lagged relevant implementations
        of the AMUSE-Algorithm. Especially it is implemented the computation of the
        time-lagged covariance matrix :math:`C^{\\tau}`.
        The class :class:`.TicaPrinCompTimeLagged` also performs the transformations, which are leads to the
        independent components :math:`Z`.

        :param i_ticaPrinComp: A :class:`.TicaPrinComp` object as input parameter to indicate the dependencies.
        :type i_ticaPrinComp: :class:`.TicaPrinComp`
        """

        def __init__(self, i_ticaPrinComp):

            self.m_prinComp = i_ticaPrinComp
            self.param_timeLag = self.m_prinComp.param_timeLag
            self.m_covMatTimeLag = np.array([])
            self.m_covMatTimeLagSym = np.array([])
            self.m_eigenDecomp = ticaEDecomp.TicaEigenDecomp(None)

        #---------------------------------------------------------------------------------------------#
        def computeCovariance(self):
            """
            Computes the time-lagged covariance matrix :math:`C^{\\tau}` with
            :math:`c_{ij}^{\\tau} = \\frac{1}{N-\\tau-1}\\sum_{t=1}^{N-\\tau}x_{it}x_{jt+\\tau}`
            """

            self.m_prinComp.m_dataImporter.rewind()

            dataChunk = np.asarray(self.m_prinComp.m_dataImporter.get_data(self.m_prinComp.param_chunkSize), dtype=np.float32)
            dimData = self.m_prinComp.m_dataImporter.get_shape_inFile()
            shapeDataChunk = dataChunk.shape
            m = shapeDataChunk[0]

            self.m_covMatTimeLag = matlib.zeros([shapeDataChunk[1], shapeDataChunk[1]], dtype = np.float64)
            normalizedPCs = matlib.zeros([shapeDataChunk[1], shapeDataChunk[1]], dtype = np.float64)

            if 1 < m:

                pcs = self.m_prinComp.computePCs(dataChunk, shapeDataChunk[1])
                del dataChunk
                normalizedPCs = self.m_prinComp.normalizePCs(pcs)
                del pcs
                self.m_covMatTimeLag += np.dot(normalizedPCs[0:(m - self.param_timeLag), :].T,
                                               normalizedPCs[self.param_timeLag:m, :])

            lastRowsChunkBefore = normalizedPCs[(m-self.param_timeLag):m, :]
            del normalizedPCs

            while self.m_prinComp.m_dataImporter.has_more_data():

                dataChunk = np.asarray(self.m_prinComp.m_dataImporter.get_data(self.m_prinComp.param_chunkSize), dtype=np.float32)
                shapeDataChunk = dataChunk.shape
                m = shapeDataChunk[0]

                pcs = self.m_prinComp.computePCs(dataChunk, shapeDataChunk[1])
                del dataChunk
                normalizedPCs = self.m_prinComp.normalizePCs(pcs)
                del pcs
                self.m_covMatTimeLag += np.dot(matlib.asmatrix(lastRowsChunkBefore).T,
                                               matlib.asmatrix(normalizedPCs[0:self.param_timeLag, :]))

                if 1 < m:

                    self.m_covMatTimeLag += np.dot(normalizedPCs[0:(m - self.param_timeLag), :].T,
                                                   normalizedPCs[self.param_timeLag:m, :])

                lastRowsChunkBefore = normalizedPCs[(m-self.param_timeLag):m, :]
                del normalizedPCs

            self.m_covMatTimeLag *= 1.0 / (dimData[0] - self.param_timeLag - 1.0)

        #---------------------------------------------------------------------------------------------#
        def symmetrizeCovariance(self):
            """
            Symmetrizes the time-lagged covariance matrix :math:`C^{\\tau}` by
            :math:`C_{sym}^{\\tau} = \\frac{1}{2} \\left[ C^{\\tau} + \\left( C^{\\tau} \\right)^{T} \\right]`
            """

            self.m_covMatTimeLagSym = 0.5 * ( self.m_covMatTimeLag + self.m_covMatTimeLag.transpose() )

        #---------------------------------------------------------------------------------------------#
        def setTimeLag(self, i_timeLag):
            """
            Set a new time-lag value.
            :param i_timeLag:
            :type: int
            """

            self.param_timeLag = i_timeLag

        #---------------------------------------------------------------------------------------------#
        def performTransformation(self, i_domComp):
            """
            Computes the independent components and saves needed components in a output numpy binary file.
            :param i_domComp: Needed components of TICA
            :type i_domComp: int
            """

            outFileShape = [self.m_prinComp.m_dataImporter.get_shape_inFile()[0], i_domComp]
            self.m_prinComp.m_dataImporter.create_out_file(self.m_prinComp.param_outFileName, outFileShape)

            self.m_prinComp.m_dataImporter.rewind()

            dataChunk = np.asarray(self.m_prinComp.m_dataImporter.get_data(self.m_prinComp.param_chunkSize), dtype=np.float32)
            shapeDataChunk = dataChunk.shape

            pcs = self.m_prinComp.computePCs(dataChunk, shapeDataChunk[1])
            del dataChunk
            normalizedPCs = self.m_prinComp.normalizePCs(pcs)
            del pcs
            ics = np.dot(normalizedPCs, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp])
            del normalizedPCs

            self.m_prinComp.m_dataImporter.write_data(ics)
            del ics

            while self.m_prinComp.m_dataImporter.has_more_data():

                dataChunk = np.asarray(self.m_prinComp.m_dataImporter.get_data(self.m_prinComp.param_chunkSize), dtype=np.float32)
                shapeDataChunk = dataChunk.shape

                pcs = self.m_prinComp.computePCs(dataChunk, shapeDataChunk[1])
                del dataChunk
                normalizedPCs = self.m_prinComp.normalizePCs(pcs)
                del pcs
                ics = np.dot(normalizedPCs, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp])
                del normalizedPCs

                self.m_prinComp.m_dataImporter.write_data(ics)
                del ics

        #---------------------------------------------------------------------------------------------#
        def computeICs(self, i_numDomComp = 1):
            '''
            Main method in the class :class:`.TicaPrinCompTimeLagged`.
            Computes the independent components on the basis of normalized principle components supplied by
            :class:`.TicaPrinComp`.

            :param i_numDumComp: Number of needed independent components.
            :type i_numDomComp: int
            '''

            self.computeCovariance()
            self.symmetrizeCovariance()
            self.m_eigenDecomp.computeEigenDecomp(self.m_covMatTimeLagSym)

            # domComp = self.m_eigenDecomp.m_eigenVecReal.shape[1]
            # if 1 > i_amountOfTotalVariance:
            #
            #     domComp = self.m_prinComp.calcNumbOfDomComps( i_amountOfTotalVariance )

            self.performTransformation(i_numDomComp)






                

