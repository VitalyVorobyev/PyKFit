""" """

from fitbase import FitBase
from particle import Particle

import numpy as np
from scipy.linalg import block_diag

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
        lo, hi = 0, 2
        jlo, jhi = 0, 2
        for _ in range(len(self.helixes)):
            d0, phi0, omega, z0, tanl = self.state['al0'][jlo:jhi]
            vtxx, vtxy, vtxz = self.state['al0'][-3:]
            dztan = (vtxz - z0) / tanl
            phiv = phi0 + omega * dztan

            sphiv, cphiv = np.sin(phiv), np.cos(phiv)
            sphi0, cphi0 = np.sin(phi0), np.cos(phi0)
            d0om = 1 + d0 * omega

            # offset
            self.state['d'][lo:hi] = np.array([
                vtxx - ( sphiv - d0om * sphi0),
                vtxy - (-cphiv + d0om * cphi0),
            ]) / omega

            # gradient
            # 2x6 block
            self.state['D'][lo:hi, jlo:jhi] = np.array([
                [
                    sphi0,
                    (-cphiv + d0om * cphi0) / omega,
                    (sphiv - sphi0) / omega**2 - dztan * cphiv,
                    # TODO
                ],
                [
                    -cphi0,
                    (-sphiv + d0om * sphi0) / omega,
                    (-cphiv + cphi0) / omega**2 - dztan * sphiv,
                    # TODO
                ]
            ])
            
            lo, hi = hi, hi + 2
            jlo, jhi = jhi, jhi + self.trksize
