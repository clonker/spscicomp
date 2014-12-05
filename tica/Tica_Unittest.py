__author__ = 'JoschkaB'

import numpy as np
import unittest
from Tica_PrincipleComp import TicaPrinComp
from Tica_PrincipleComp import TicaPrinCompTimeLagged
from Tica_Amuse import TicaAmuse

##testdata##
data = np.array(((1,6),(2,2)))
blub = TicaPrinComp(data)
blub.computePC()
blub.normalizPC()

##testdata TimeLagged##
data2 = np.array(((3,4),(6,2)))
blub2 = TicaPrinComp(data2)
blub2.computePC()
blub2.normalizPC()

####
blubTL = TicaPrinCompTimeLagged(blub.m_pcNorm,blub2.m_pcNorm)
blubTL.computePC()

class TicaUnittest(unittest.TestCase):

    def testPrincipleComp1(self):
        self.assertEqual(blub.m_covMat.tolist(), np.array(((0.5,-2),(-2,8))).tolist())
        self.assertEqual(blub.m_dataMeanFree.tolist(), np.array(((-0.5,2),(0.5,-2))).tolist())
        self.assertEqual(blub.m_pc.tolist(), np.array(((-2.0615527629852295,0),(2.0615527629852295,0))).tolist())
        self.assertEqual(blub.m_pcNorm.tolist(), np.array(((-0.7071026563644409,0),(0.7071026563644409,0))).tolist())

        self.assertEqual(blub.m_eigenDecomp.m_eigenVal.tolist(), np.array((0,8.5)).tolist())
        self.assertAlmostEqual(blub.m_eigenDecomp.m_eigenVec.all(), np.array(((-0.970143,-0.242536),(-0.242536,0.970143))).all())

    def testPrincipleComp2(self):
        self.assertEqual(blub2.m_covMat.tolist(), np.array(((4.5,-3),(-3,2))).tolist())
        self.assertEqual(blub2.m_dataMeanFree.tolist(), np.array(((-1.5,1),(1.5,-1))).tolist())
        self.assertEqual(blub2.m_pc.tolist(), np.array(((-1.8027756214141846,0),(1.8027756214141846,0))).tolist())
        self.assertEqual(blub2.m_pcNorm.tolist(), np.array(((-0.7071013450622559,0),(0.7071013450622559,0))).tolist())

        self.assertEqual(blub2.m_eigenDecomp.m_eigenVal.tolist(), np.array((6.5,0)).tolist())
        self.assertAlmostEqual(blub2.m_eigenDecomp.m_eigenVec.all(), np.array(((-0.83205,0.5547),(-0.5547,-0.83205))).all())

    def testPrinCompTimeLagged(self):
        self.assertEqual(blubTL.m_covMatTimeLag.tolist(), np.array(((0.999986469745636,0),(0,0))).tolist())
        self.assertEqual(blubTL.m_ic.tolist(), np.array(((-0.7071026563644409,0),(0.7071026563644409,0))).tolist())
        self.assertEqual(blubTL.m_covMatTimeLagSym.tolist(), np.array(((0.999986469745636,0),(0,0))).tolist())