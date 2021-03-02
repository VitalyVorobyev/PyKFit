""" """

import numpy as np
from scipy.linalg import block_diag

from .fitbase import FitBase
from .particle import Particle
from .helix import vertexOffset, vertexOffsetGradient, Helix

class VertexFit(FitBase):
    def __init__(self, v0, v0err, tracks=[]):
        super().__init__(2, 10)
        self.tracks = tracks
        self.trksize = 5  # helix parameters
        self.descendants_updated = False
        self.v0 = v0
        self.v0err = v0err
    
    def numConstraints(self):
        return 2 * len(self.tracks)

    def numParams(self):
        return self.state['al0'].size

    def helixPars(self, idx):
        return self.state['al1'][idx * self.trksize:(idx + 1) * self.trksize]

    def helixErrs(self, idx):
        lo, hi = idx * self.trksize, (idx + 1) * self.trksize
        return self.state['Val1'][lo:hi, lo:hi]

    def vertex(self):
        return self.state['al1'][-3:]

    def vertexError(self):
        return self.state['Val1'][-3:,-3:]

    def fillInputMatrix(self):
        self.state.update({
            'al0' : np.concatenate([trk.helix.pars for trk in self.tracks] + [self.v0]).reshape(-1, 1),
            'Val0' : block_diag(*[trk.helix.errmtx for trk in self.tracks], self.v0err)
        })

        self.state['d'] = np.empty(self.numConstraints()).reshape(-1, 1)
        self.state['D'] = np.zeros((self.numParams(), self.numConstraints()))

    def calculateGradients(self):
        vtx = self.state['al0'][-3:]

        lo, hi = 0, 2
        jlo, jhi = 0, self.trksize
        for _ in range(len(self.tracks)):
            hpars = self.state['al0'][jlo:jhi]

            self.state['d'][lo:hi] = vertexOffset(hpars, vtx)
            hparsGrad, vtxGrad = vertexOffsetGradient(hpars, vtx)
            self.state['D'][jlo:jhi, lo:hi] = hparsGrad
            self.state['D'][-3:, lo:hi] = vtxGrad

            lo, hi = hi, hi + 2
            jlo, jhi = jhi, jhi + self.trksize

    def updateDescendants(self):
        vtx = self.vertex()
        for idx, trk in enumerate(self.tracks):
            trk.helix = Helix(*self.helixPars(idx), self.helixErrs(idx))
            trk.buildCartesian(trk.helix.lengthAtZ(vtx[2]))

    def makeParent(self):
        if not self.descendants_updated:
            self.updateDescendants()
        
        errmtx = sum([trk.errmtx for trk in self.tracks])
        errmtx[:3, :3] = self.vertexError()
        errmtx[-3:, :3] = errmtx[:3, -3:] = np.zeros((3, 3))
        return Particle(
            charge=sum(trk.charge for trk in self.tracks),
            momentum=sum([trk.momentum for trk in self.tracks]),
            position=self.vertex(),
            errmtx=errmtx,
            extra={'chisq':  self.chisq}
        )
