__author__ = 'rickwg'

import numpy as np
import numpy.matlib as matlib
import Tica_EigenDecomp as ticaEDecomp
from common_data_importer import CommonBinaryFileDataImporter
#import ticaC
import os


class TicaPrinComp:
  """Naive class for computing principle components(PCs)."""

  def __init__( self, i_inFileName = None, i_outFileName = "../testdata/tica_tempOutput.npy", i_addEpsilon = 1e-16, i_timeLag = 1 ):

    self.param_fileSizeThreshold = 500*1e+6 # file size in byte
    self.param_outFileName       = i_outFileName

    if i_inFileName is not None:

      self.m_fileSize     = os.path.getsize( i_inFileName )
      self.m_dataImporter = CommonBinaryFileDataImporter( i_inFileName )
      self.m_dataImporter.create_out_file( self.param_outFileName )

      self.param_addEpsilon = i_addEpsilon
      self.param_chunkSize  = 1000#None
      self.param_timeLag    = i_timeLag
      self.computeChunkSize( )
      print(self.param_chunkSize)
      self.m_covMat      = np.array( [] )
      self.m_eigenDecomp = ticaEDecomp.TicaEigenDecomp( None )
      self.m_colMeans    = None

    else:

      self.m_eigenDecomp = ticaEDecomp.TicaEigenDecomp( None )

  # ---------------------------------------------------------------------------------------------#

  def getPrinCompTL( self ):

    return self.TicaPrinCompTimeLagged( self )

  # ---------------------------------------------------------------------------------------------#
  def getTempFileName( self ):

    return self.param_outFileName

  # ---------------------------------------------------------------------------------------------#
  def computeChunkSize( self ):

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
  def computeColMeans( self ):

    self.m_dataImporter.rewind( )

    dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
    dataShape = dataChunk.shape
    dimData   = dataShape[0]
    sumCol    = np.sum( dataChunk[:, 0:dataShape[1]], dtype = np.float32, axis = 0 )

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
      dimData  += len( dataChunk )
      sumCol   += np.sum( dataChunk[:, 0:dataShape[1]], dtype = np.float32, axis = 0 )

    self.m_colMeans = 1.0 / dimData * sumCol


  # ---------------------------------------------------------------------------------------------#
  def makeDataMeanFree( self ):

    self.computeColMeans( )
    self.m_dataImporter.rewind( )

    dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
    dataShape = dataChunk.shape
    meanFreeChunk = dataChunk[:, 0:dataShape[1]] - self.m_colMeans
    self.m_dataImporter.write_data( meanFreeChunk )

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
      meanFreeChunk = dataChunk[:, 0:dataShape[1]] - self.m_colMeans
      self.m_dataImporter.write_data( meanFreeChunk )

  # ----------------------------------------------------------------------------------------------#
  def computeCovariance( self ):

    self.computeColMeans( )
    self.m_dataImporter.rewind( )

    dataShape = self.m_dataImporter.get_shape_inFile( )
    #covMat = ticaC.computeCov(self.m_dataImporter.get_data,self.m_dataImporter.has_more_data,self.param_chunkSize,dataShape[0])

    dataChunk     = self.m_dataImporter.get_data( self.param_chunkSize )
    meanFreeChunk = dataChunk[:, 0:dataShape[1]] - self.m_colMeans
    cov           = np.dot( meanFreeChunk.T, meanFreeChunk )

    while self.m_dataImporter.has_more_data( ):

      dataChunk     = self.m_dataImporter.get_data( self.param_chunkSize )
      meanFreeChunk = dataChunk[:, 0:dataShape[1]] - self.m_colMeans
      cov          += np.dot( meanFreeChunk.T, meanFreeChunk )

    self.m_covMat = 1.0 / (dataShape[0] - 1.0) * cov

  #---------------------------------------------------------------------------------------------#
  def computePCs( self, i_dataChunk, i_domComp ):

    if not 1 < i_dataChunk.shape[0]:

      meanFreeChunk = i_dataChunk[:, 0:i_dataChunk.shape[1]] - self.m_colMeans
      pc = np.dot( matlib.asmatrix(meanFreeChunk), self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )

    else:

      meanFreeChunk = i_dataChunk[:, 0:i_dataChunk.shape[1]] - self.m_colMeans
      pc = np.dot(  meanFreeChunk, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )

    return pc

  #---------------------------------------------------------------------------------------------#
  def computePC( self, i_amountOfTotalVariance = 1.0 ):

    self.makeDataMeanFree()
    self.computeCovariance( )
    self.m_eigenDecomp.computeEigenDecomp( self.m_covMat )

    domComp = self.m_eigenDecomp.m_eigenVecReal.shape[1]
    if 1 > i_amountOfTotalVariance:

      domComp = self.calcNumbOfDomComps( i_amountOfTotalVariance )

    self.performTransformation( domComp )

  #---------------------------------------------------------------------------------------------#
  def computeEigenDecompCov( self ):

      self.m_eigenDecomp.computeEigenDecomp( self.m_covMat )

  #---------------------------------------------------------------------------------------------#
  def normalizePCs( self, i_pcs ):

    lamb = 1.0 / np.sqrt( self.m_eigenDecomp.m_eigenValReal + self.param_addEpsilon )

    if not 1 < i_pcs.shape[0]:

      pcNorm = np.dot( matlib.asmatrix(i_pcs), np.diag( lamb ) )

    else:

      pcNorm = np.dot( i_pcs, np.diag( lamb ) )

    return pcNorm

  #---------------------------------------------------------------------------------------------#
  def calcNumbOfDomComps( self, i_amountOfTotalVariance ):

    totalVariance = np.sum( self.m_eigenDecomp.m_eigenValReal )
    sumEigVal = 0
    for i, e in enumerate( self.m_eigenDecomp.m_eigenValReal ):

      sumEigVal += e
      if sumEigVal / totalVariance >= i_amountOfTotalVariance:

        return i + 1

  #---------------------------------------------------------------------------------------------#
  def getOutFileName( self ):

    return self.param_outFileName

###################################################################################################

  class TicaPrinCompTimeLagged:
    """Naive class for computing principle components(PCs)."""

    def __init__( self, i_ticaPrinComp ):

      self.m_prinComp         = i_ticaPrinComp
      self.param_timeLag      = self.m_prinComp.param_timeLag
      self.m_covMatTimeLag    = np.array( [] )
      self.m_covMatTimeLagSym = np.array( [] )
      self.m_eigenDecomp      = ticaEDecomp.TicaEigenDecomp( None )

    #---------------------------------------------------------------------------------------------#
    def computeCovariance( self ):

      self.m_prinComp.m_dataImporter.rewind( )

      dataChunk      = self.m_prinComp.m_dataImporter.get_data( self.m_prinComp.param_chunkSize )
      dimData        = self.m_prinComp.m_dataImporter.get_shape_inFile( )
      shapeDataChunk = dataChunk.shape
      m              = shapeDataChunk[0]

      self.m_covMatTimeLag = matlib.zeros( [shapeDataChunk[1], shapeDataChunk[1]],dtype=np.float64 )
      pcs                  = matlib.zeros( [shapeDataChunk[1], shapeDataChunk[1]],dtype=np.float64 )
      normalizedPCs        = matlib.zeros( [shapeDataChunk[1], shapeDataChunk[1]],dtype=np.float64 )

      if 1 < m:

        pcs                   = self.m_prinComp.computePCs( dataChunk, shapeDataChunk[1] )
        normalizedPCs         = self.m_prinComp.normalizePCs( pcs )
        self.m_covMatTimeLag += np.dot( normalizedPCs[0:(m-self.param_timeLag), :].T, normalizedPCs[self.param_timeLag:m, :] )

      lastRowChunkBefore = normalizedPCs[m-1, :]

      while self.m_prinComp.m_dataImporter.has_more_data( ):

        dataChunk      = self.m_prinComp.m_dataImporter.get_data( self.m_prinComp.param_chunkSize )
        shapeDataChunk = dataChunk.shape
        m              = shapeDataChunk[0]

        pcs                   = self.m_prinComp.computePCs( dataChunk, shapeDataChunk[1] )
        normalizedPCs         = self.m_prinComp.normalizePCs( pcs )
        self.m_covMatTimeLag += np.dot( matlib.asmatrix(lastRowChunkBefore).T, matlib.asmatrix(normalizedPCs[0, :]) )

        if 1 < m:

          self.m_covMatTimeLag += np.dot( normalizedPCs[0:(m-self.param_timeLag), :].T, normalizedPCs[self.param_timeLag:m, :] )

        lastRowChunkBefore = normalizedPCs[m-1, :]

      self.m_covMatTimeLag *= 1.0 / (dimData[0] - self.param_timeLag - 1.0)

    #---------------------------------------------------------------------------------------------#
    def symmetrizeCovariance( self ):

      self.m_covMatTimeLagSym = 0.5 * ( self.m_covMatTimeLag + self.m_covMatTimeLag.transpose( ) )

    #---------------------------------------------------------------------------------------------#
    def setTimeLag( self, i_timeLag ):

      self.param_timeLag = i_timeLag

    #---------------------------------------------------------------------------------------------#
    def performTransformation( self, i_domComp ):

      self.m_prinComp.m_dataImporter.rewind( )

      dataChunk      = self.m_prinComp.m_dataImporter.get_data( self.m_prinComp.param_chunkSize )
      shapeDataChunk = dataChunk.shape

      pcs            = self.m_prinComp.computePCs( dataChunk, shapeDataChunk[1] )
      normalizedPCs  = self.m_prinComp.normalizePCs( pcs )
      ics            = np.dot( normalizedPCs, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )

      self.m_prinComp.m_dataImporter.write_data( ics )

      while self.m_prinComp.m_dataImporter.has_more_data( ):

        dataChunk      = self.m_prinComp.m_dataImporter.get_data( self.m_prinComp.param_chunkSize )
        shapeDataChunk = dataChunk.shape

        pcs            = self.m_prinComp.computePCs( dataChunk, shapeDataChunk[1] )
        normalizedPCs  = self.m_prinComp.normalizePCs( pcs )
        ics            = np.dot( normalizedPCs, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )

        self.m_prinComp.m_dataImporter.write_data( ics )

    #---------------------------------------------------------------------------------------------#
    def computeICs( self, i_amountOfTotalVariance = 1 ):

      self.computeCovariance(  )
      self.symmetrizeCovariance( )
      self.m_eigenDecomp.computeEigenDecomp( self.m_covMatTimeLagSym )

      domComp = self.m_eigenDecomp.m_eigenVecReal.shape[1]
      if 1 > i_amountOfTotalVariance:

        domComp = self.m_prinComp.calcNumbOfDomComps( i_amountOfTotalVariance )

      self.performTransformation( domComp )






                

