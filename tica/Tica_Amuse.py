__author__ = 'rickwg'

import numpy as np
import Tica_PrincipleComp as ticaPrinComp

class TicaAmuse:

    def __init__(self, i_dataInstantaneous, i_dataTimeLagged):

        if 0 < i_dataInstantaneous.shape[0] and 0 < i_dataTimeLagged.shape[0]:

            self.m_dataInstant     = np.asmatrix(i_dataInstantaneous, dtype=np.float32)
            self.m_dataTimeLag     = np.asmatrix(i_dataTimeLagged, dtype=np.float32)
            self.m_pcInstant       = ticaPrinComp.TicaPrinComp(i_dataInstantaneous)
            self.m_pcTimeLag       = ticaPrinComp.TicaPrinComp(i_dataTimeLagged)
            self.m_icAmuse         = ticaPrinComp.TicaPrinCompTimeLagged(None, None)

    def performAmuse(self):

        self.m_pcInstant.computePC()
        self.m_pcInstant.normalizPC()
        self.m_pcTimeLag.computePC()
        self.m_pcTimeLag.normalizPC()

        self.m_icAmuse.__init__(self.m_pcInstant.m_pcNorm, self.m_pcTimeLag.m_pcNorm)
        self.m_icAmuse.computePC()