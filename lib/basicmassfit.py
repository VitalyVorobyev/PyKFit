""" """

from .fitbase import FitBase
from .particle import Particle
from .auxkine import massSqFromP3E, momentumFromMass, make_beta

import numpy as np
from scipy.linalg import block_diag

class BasicMassFit(FitBase):
    def __init__(self, targetMass):
        super().__init__(2, 10)
        self.targetMass = targetMass
        self.tracks = []
        self.trksize = 3
        self.descendants_updated = False

    def numParams(self):
        return self.state['al0'].size

    def numConstraints(self):
        return 1

    def fillInputMatrix(self):
        assert len(self.tracks) >= self.necessaryTrackCount

        self.state.update({
            'al0' : np.concatenate([trk.threeMomentum for trk in self.tracks]).reshape(-1, 1),
            'Val0': block_diag(*[trk.threeMomentumError for trk in self.tracks])
        })
        self.state['D'] = np.empty(self.state['al0'].shape)

    def updateDescendants(self):
        ilo, ihi = 0, self.trksize
        for trk in self.tracks:
            p3 = self.state['al1'][ilo:ihi]
            p4 = momentumFromMass(p3, trk.mass)
            beta = make_beta(p4)

            p3err = self.state['Val1'][ilo:ihi, ilo:ihi]
            p4err = np.empty((4,4))
            p4err[:-1,:-1] = p3err
            p4err[:3, 3:4] = p3err @ beta  # [3x3] x [3x1]
            p4err[3:4, :3] = p4err[3, :3].T
            p4err[3, 3] = beta.T @ p3err @ beta

            trk.momentum = p4
            trk.momentumError = p4err

            ilo, ihi = ihi, ihi + self.trksize
        self.descendants_updated = True

    def calculateGradients(self):
        al1_sum = np.zeros((self.trksize, 1))
        energy = []
        ilo, ihi = 0, self.trksize
        for trk in self.tracks:
            energy.append(np.sqrt(trk.mass**2 + np.sum(self.state['al1'][ilo:ihi]**2)))
            al1_sum += self.state['al1'][ilo:ihi]
            ilo, ihi = ihi, ihi + self.trksize

        totalEnergy = sum(energy)
        self.state['d'] = np.array(massSqFromP3E(al1_sum.ravel(), totalEnergy) - self.targetMass**2).reshape(-1, 1)

        ilo, ihi = 0, self.trksize
        for trk, e in zip(self.tracks, energy):
            self.state['D'][ilo:ihi] = 2 * (totalEnergy * self.state['al1'][ilo:ihi] / e - al1_sum)
            ilo, ihi = ihi, ihi + self.trksize

    def makeParent(self):
        if not self.descendants_updated:
            self.updateDescendants()
        return Particle(
            charge=sum(trk.charge for trk in self.tracks),
            momentum=sum([trk.momentum for trk in self.tracks]),
            errmtx=sum([trk.momentumError for trk in self.tracks]),
            extra={'chisq':  self.chisq}
        )
    