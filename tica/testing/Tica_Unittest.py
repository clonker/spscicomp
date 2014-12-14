__author__ = 'JoschkaB'

import numpy as np
import unittest
from Tica_DataImport import TicaDataImport
from Tica_PrincipleComponents import TicaPrinComp


asd = TicaDataImport('C:\\Users\\Josch_000\\Downloads\\testDaten.csv')
blub = TicaPrinComp(asd.m_data)
blub.computePC()
meanFree = np.array(((0.5,1),(-0.5,-1)))
covMatrix = np.array(((0.5,1),(1,2)))

class TicaUnittest(unittest.TestCase):

    def testDataImport(self):
        self.assertEqual(TicaDataImport('C:\\Users\\Josch_000\\Downloads\\testDaten.csv').m_data.tolist(), asd.m_data.tolist())


    def testPrincipleComponents(self):
        self.assertEqual(blub.m_covMat.tolist(), covMatrix.tolist())
        self.assertEqual(blub.m_dataMeanFree.tolist(), covMatrix.tolist())
        #self.assertEqual(TicaPrinComp(asd.m_csvData).m_pc.tolist(), blub.m_data.tolist())

if __name__ == "__main__":
    unittest.main()