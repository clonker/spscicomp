__author__ = 'rickwg'

import numpy as np
import numpy.matlib as matlib
import Tica_EigenDecomp as ticaEDecomp
from common_data_importer import CommonBinaryFileDataImporter
import ticaC
import os


class TicaPrinComp:
  """Naive class for computing principle components(PCs)."""

  def __init__( self, i_fileName = None, i_outFileName = "../testdata/tica_tempOutPut.npy", i_addEpsilon = 1e-16 ):

    self.param_fileSizeThreshold = 1  # 200*1e+6 # file size in byte
    self.param_outFileName = i_outFileName

    if i_fileName is not None:

      self.m_fileSize = os.path.getsize( i_fileName )
      self.m_dataImporter = CommonBinaryFileDataImporter( i_fileName )
      self.m_dataImporter.create_out_file( self.param_outFileName )

      self.param_addEpsilon = i_addEpsilon
      self.param_chunkSize = 1# None
      #self.computeChunkSize( )

      self.m_covMat = np.array( [] )
      self.m_eigenDecomp = ticaEDecomp.TicaEigenDecomp( None )
      self.m_colMeans = None

    else:

      self.m_eigenDecomp = ticaEDecomp.TicaEigenDecomp( None )

  # ---------------------------------------------------------------------------------------------#
  def getTempFileName( self ):

    return self.param_outFileName

  # ---------------------------------------------------------------------------------------------#
  def computeChunkSize( self ):

    if self.m_fileSize < self.param_fileSizeThreshold:

      self.param_chunkSize = len( self.m_dataImporter._file )

    else:

      fileSize = self.m_fileSize

      if fileSize % 2 is not 0:

        fileSize += 1

      fileSize /= 2

      dimData = len( self.m_dataImporter._file )
      if dimData % 2 is not 0:

        self.param_chunkSize = ( dimData + 1 ) / 2

      else:

        self.param_chunkSize = dimData / 2

      while fileSize > self.param_fileSizeThreshold:

        self.param_chunkSize /= 2
        fileSize /= 2

  # ---------------------------------------------------------------------------------------------#
  def computeChunkMeans( self ):

    self.m_dataImporter.rewind( )

    dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
    dataShape = dataChunk.shape
    dimData = dataShape[0]
    sumCol = np.sum( dataChunk[:, 0:dataShape[1]], dtype = np.float32, axis = 0 )

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
      dimData += len( dataChunk )
      sumCol += np.sum( dataChunk[:, 0:dataShape[1]], dtype = np.float32, axis = 0 )

    self.m_colMeans = 1.0 / dimData * sumCol


  # ---------------------------------------------------------------------------------------------#
  def makeDataMeanFree( self ):

    self.computeChunkMeans( )
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

    self.m_dataImporter.rewind( )
    dataChunk = self.m_dataImporter.get_outData( self.param_chunkSize )
    dimData = self.m_dataImporter.get_shape_outFile( )[0]
    cov = np.dot( dataChunk.T, dataChunk )

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_outData( self.param_chunkSize )
      cov += np.dot( dataChunk.T, dataChunk )

    self.m_covMat = 1.0 / (dimData - 1.0) * cov

  #---------------------------------------------------------------------------------------------#
  def performTransformation( self, i_domComp ):

    self.m_dataImporter.rewind( )

    dataChunk = self.m_dataImporter.get_outData( self.param_chunkSize )
    pc = np.dot( dataChunk, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )
    self.m_dataImporter.write_data( pc )

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_outData( self.param_chunkSize )
      pc = np.dot( dataChunk, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )
      self.m_dataImporter.write_data( pc )

  #---------------------------------------------------------------------------------------------#
  def computePC( self, i_amountOfTotalVariance = 1.0 ):

    self.makeDataMeanFree( )
    self.computeCovariance( )
    self.m_eigenDecomp.computeEigenDecomp( self.m_covMat )

    domComp = self.m_eigenDecomp.m_eigenVecReal.shape[1]
    if 1 > i_amountOfTotalVariance:

      domComp = self.calcNumbOfDomComps( i_amountOfTotalVariance )

    self.performTransformation( domComp )

  #---------------------------------------------------------------------------------------------#
  def normalizPC( self ):

    self.m_dataImporter.rewind( )

    dataChunk = self.m_dataImporter.get_outData( self.param_chunkSize )
    lamb = 1.0 / np.sqrt( self.m_eigenDecomp.m_eigenValReal + self.param_addEpsilon )
    pcNorm = np.dot( dataChunk, np.diag( lamb ) )
    self.m_dataImporter.write_data( pcNorm )

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_outData( self.param_chunkSize )
      pcNorm = np.dot( dataChunk, np.diag( lamb ) )
      self.m_dataImporter.write_data( pcNorm )

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

class TicaPrinCompTimeLagged( TicaPrinComp ):
  """Naive class for computing principle components(PCs)."""

  def __init__( self, i_fileName = None, i_outFileName = None, i_timeLag = 1 ):

    super().__init__( i_fileName = i_fileName, i_outFileName = i_outFileName )

    self.param_timeLag = i_timeLag
    self.m_covMatTimeLag = np.array( [] )
    self.m_covMatTimeLagSym = np.array( [] )

  #---------------------------------------------------------------------------------------------#
  def computeCovariance( self ):

    self.m_dataImporter.rewind( )
    dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
    dimData = self.m_dataImporter.get_shape_inFile( )[0]
    dataShape = dataChunk.shape
    m = dataShape[0]

    if 1 < m:

      cov = np.dot( dataChunk[0:m-self.param_timeLag, :].T, dataChunk[self.param_timeLag:m, :] )

    else:

      cov = matlib.zeros( [dataShape[1], dataShape[1]],dtype=np.float32 )

    lastRowChunkBefore = dataChunk[m-1, :]

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
      dataShape = dataChunk.shape
      m = dataShape[0]
      cov += np.dot( lastRowChunkBefore.T, dataChunk[0, :] )

      if 1 < m:

        cov += np.dot( dataChunk[0:m-self.param_timeLag, :].T, dataChunk[self.param_timeLag:m, :] )

      lastRowChunkBefore = dataChunk[m-1, :]

    self.m_covMatTimeLag = 1.0 / (dimData - self.param_timeLag - 1.0) * cov

  #---------------------------------------------------------------------------------------------#
  def symmetrizeCovariance( self ):

    self.m_covMatTimeLagSym = 0.5 * ( self.m_covMatTimeLag + self.m_covMatTimeLag.transpose( ) )

  #---------------------------------------------------------------------------------------------#
  def setTimeLag( self, i_timeLag ):

    self.param_timeLag = i_timeLag

  #---------------------------------------------------------------------------------------------#
  def performTransformation( self, i_domComp ):

    self.m_dataImporter.rewind( )

    dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
    pc = np.dot( dataChunk, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )
    self.m_dataImporter.write_data( pc )

    while self.m_dataImporter.has_more_data( ):

      dataChunk = self.m_dataImporter.get_data( self.param_chunkSize )
      pc = np.dot( dataChunk, self.m_eigenDecomp.m_eigenVecReal[:, 0:i_domComp] )
      self.m_dataImporter.write_data( pc )

  #---------------------------------------------------------------------------------------------#
  def computePC( self, i_amountOfTotalVariance = 1 ):

    # self.m_data = self.makeDataMeanFree(self.m_data)
    self.computeCovariance( )
    self.symmetrizeCovariance( )
    self.m_eigenDecomp.computeEigenDecomp( self.m_covMatTimeLagSym )

    domComp = self.m_eigenDecomp.m_eigenVecReal.shape[1]
    if 1 > i_amountOfTotalVariance:

      domComp = self.calcNumbOfDomComps( i_amountOfTotalVariance )

    self.performTransformation( domComp )






                

