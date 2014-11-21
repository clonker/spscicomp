__author__ = 'rickwg'

import csv
import numpy as np


class TicaDataImport:

    def __init__(self, i_fileName):

        self.m_data = []
        self.readTable(i_fileName)

    # ---------------------------------------------------------------------------------------------#
    def readTable(self, i_fileName):

        # try:
            self.m_data = np.loadtxt(i_fileName, delimiter=';')
            # print(type(self.m_csvData))

        # except IOError as ioErr:
        #
        #     print('cannot open'.format(ioErr))

    def getData(self):

        return np.asmatrix(self.m_data, dtype=np.float32)
