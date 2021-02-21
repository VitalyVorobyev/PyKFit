""" """

from numpy.core.numeric import full
from fitbase import FitBase

import numpy as np
from scipy.linalg import block_diag

def momentumFromMass(p3, mass):
    return np.array(list(p3) + [np.sqrt(mass**2 + np.sum(p3**2))])

def make_beta(p):
    return p[:3].reshape(-1, 1) / p[3]

def massSqFromP3E(p3, e):
    return e**2 - np.sum(p3**2)

def massSqFromP4(p4):
    return massSqFromP3E(p4[:3], p4[3])

class BasicMassFit(FitBase):
    def __init__(self, targetMass):
        self.targetMass = targetMass
        self.necessaryTrackCount = 2
        self.tracks = []
        self.trksize = 3
        self.descendants_updated = False

    def fillInputMatrix(self):
        assert len(self.tracks) >= self.necessaryTrackCount

        self.state.update({
            'al0' : np.concatenate([trk.momentum for trk in self.tracks]).reshape(-1, 1),
            'Val0': block_diag(*[trk.momentumError for trk in self.tracks])
        })

    def updateDescendants(self):
        ilo, ihi = 0, self.trksize
        for trk in self.tracks:
            p3 = self.state['al1'][ilo:ihi]
            p4 = momentumFromMass(p3, trk.mass)
            beta = make_beta(p4)

            p3err = self.state['Val1'][ilo:ihi, ilo:ihi]
            p4err = np.zeros((4,4))
            p4err[:-1,:-1] = p3err
            p4err[3, :3] = p4err[:3, 3] = beta @ p3err
            p4err[3, 3] = beta @ p3err @ beta.T()

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
        self.state['D'] = np.empty(self.state['al0'].shape)

        ilo, ihi = 0, self.trksize
        for trk, e in zip(self.tracks, energy):
            self.state['D'][ilo:ihi] = 2 * (totalEnergy * self.state['al1'][ilo:ihi] / e - al1_sum),
            ilo, ihi = ihi, ihi + self.trksize

    def makeParent(self):
        if not self.descendants_updated:
            self.updateDescendants()
        parent_momentum = sum([trk.fourMomentum for trk in self.tracks])
        parent_momentum_error = sum([trk.momentumError for trk in self.tracks])
