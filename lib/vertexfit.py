""" """

from fitbase import FitBase
from particle import Particle

import numpy as np
from scipy.linalg import block_diag

from .helix import vertexOffset, vertexOffsetGradient

class VertexFit(FitBase):
    def __init__(self, v0, v0err, helixes=[]):
        super().__init__(2, 10)
        self.helixes = helixes
        self.trksize = 5  # helix parameters
        self.descendants_updated = False
        self.v0 = v0
        self.v0err = v0err
    
    def numConstraints(self):
        return 2 * len*(self.helixes)

    def numParams(self):
        return self.state['al0'].size

    def fillInputMatrix(self):
        self.state.update({
            'al0' : np.concatenate([h.pars for h in self.helixes] + [self.v0]),
            'Val0' : block_diag(*[h.errmtx for h in self.helixes], self.v0err)
        })
        
        self.state['d'] = np.empty(self.numConstraints())
        self.state['D'] = np.zeros((self.numConstraints(), self.numParams()))

    def calculateGradients(self):
        vtx = self.state['al0'][-3:]

        lo, hi = 0, 2
        jlo, jhi = 0, 2
        for _ in range(len(self.helixes)):
            hpars = self.state['al0'][jlo:jhi]

            self.state['d'][lo:hi] = vertexOffset(hpars, vtx)
            hparsGrad, vtxGrad = vertexOffsetGradient(hpars, vtx)
            self.state['D'][lo:hi, jlo:jhi] = hparsGrad
            self.state['D'][lo:hi, -3:] = vtxGrad

            lo, hi = hi, hi + 2
            jlo, jhi = jhi, jhi + self.trksize

