__author__ = 'rickwg'

import numpy as np
import Tica_PrincipleComp as ticaPrinComp
import numpy.matlib as matlib

class TicaAmuse:

    def __init__(self, i_data, i_timeLag = 1, i_addEps = 1e-16):


        if  i_data is not None and 0 < i_data.shape[0] and i_timeLag is not None:

            self.m_timeLag         = i_timeLag
            self.m_data            = np.asmatrix(i_data, dtype=np.float32)
            self.m_prinCompInst    = ticaPrinComp.TicaPrinComp(self.m_data, i_addEps)
            self.m_prinCompTL      = ticaPrinComp.TicaPrinCompTimeLagged(None)
            self.m_ic              = matlib.repmat(0.0, self.m_data.shape[1], self.m_data.shape[1])

    def performAmuse(self, i_thresholdICs):

        if np.isscalar(self.m_timeLag):

            self.m_prinCompInst.computePC()
            self.m_prinCompInst.normalizPC()

            self.m_prinCompTL.setData(self.m_prinCompInst.getNormalizedPCs())
            self.m_prinCompTL.computePC(i_thresholdICs)

            self.m_ic = self.m_prinCompTL.getPCsTimeLag()

        else:
            for i in self.m_timeLag :

                self.m_prinCompInst.computePC()
                self.m_prinCompInst.normalizPC()

                self.m_prinCompTL.setData(self.m_prinCompInst.getNormalizedPCs())
                self.m_prinCompTL.setTimeLag(i)
                self.m_prinCompTL.computePC(i_thresholdICs)

                self.m_ic = self.m_prinCompTL.getPCsTimeLag()

        return self.m_ic
