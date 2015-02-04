import csv
import numpy as np


class TicaDataImport:
    """
    Simple class which import csv data with the :class:`numpy` function :func:`numpy.loadtxt`.
    """

    def __init__(self, i_fileName):

        self.m_data = []
        self.readTable(i_fileName)

    # ---------------------------------------------------------------------------------------------#
    def readTable(self, i_fileName):
        """
        Read a csv file.

        :param i_fileName: A filename of the csv file.
        :type i_fileName: string
        """

        try:
            self.m_data = np.loadtxt(i_fileName, delimiter=';')
            # print(type(self.m_csvData))

        except IOError as ioErr:

            print('cannot open'.format(ioErr))

    def getData(self):
        """
        Return the loaded csv data.

        :returns: Return a numpy matrix.
        :rtype: numpy.matrix
        """

        return np.asmatrix(self.m_data, dtype=np.float32)
