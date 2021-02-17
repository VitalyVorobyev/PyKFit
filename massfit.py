""" """

from fitbase import FitBase
from makeparent import MakeParent
from errormaker import make_error_type3_single, make_error_type4
from config import speedOfLight, magneticField

import numpy as np
from scipy.linalg import block_diag

def momentumFromMass(p3, mass):
    return np.array(list(p3) + [np.sqrt(mass**2 + np.sum(p3**2))])

def massSqFromP3E(p3, e):
    return e**2 - np.sum(p3**2)

def massSqFromP4(p4):
    return massSqFromP3E(p4[:3], p4[3])

class MassFit(FitBase):
    def __init__(self, target):
        super().__init__()
        self.targetMass = target
        self.necessaryTrackCount = 2
        self.fixMass = []

        self.after = {
            'vertex' : np.zeros((3, 1)),
            'vtxErr' : None,
            'trkVtxErr' : [],
        }

        self.before = {
            'vertex' : np.zeros((3, 1)),
            'vtxErr' : np.zeros((3, 3))
        }

        self.fitVertex = False
        self.atDecayPoint = True
    
    def doFit(self):
        self.doFit1()

    def prepareInputMatrix(self):
        assert len(self.tracks) >= self.necessaryTrackCount
        self.__checkFixMass()

        pars = [trk.before['mompos'] for trk in self.tracks]
        errs = [trk.before['errmtx'] for trk in self.tracks]
        if self.fitVertex:
            pars.append(self.before['vertex'])
            errs.append(self.before['vtxErr'])

        self.state.update({
            'al0' : np.concatenate(pars).reshape(-1, 1),
            'Val0': block_diag(*errs)
        })

        if self.correlations:
            self.prepareCorrelation()


    def prepareOutputMatrix(self):
        index = 0
        for trk, fixed in zip(self.tracks, self.fixMass):
            curpars = self.state['al1'][index:index+7]
            if fixed:
                curpars[:4] = momentumFromMass(curpars[:3], trk.mass(False))
            trk.after['mompos'] = curpars
            trk.after['errmtx'] = make_error_type3_single(
                curpars[:4], self.state['Val1'][index:index+7, index:index+7], fixed)
            index += 7

        if self.fitVertex:
            index, size = 0, len(self.tracks) * 7
            self.after['vertex'] = self.state['al1'][-3:]
            self.after['vtxErr'] = self.state['Val1'][-3:, -3:]

            for trk, fixed in zip(self.tracks, self.fixMass):
                hm = self.state['Val1'][size:size + 3, index:index + 7]
                self.after['trkVtxErr'].append(make_error_type4(trk.momentum(True), hm) if fixed else hm)
                index += 7
        else:
            self.after = self.before

    def correlation(self, id1, id2, after):
        if after:
            assert self.fitted

        return make_error_type4(
                self.tracks[id1].momentum(after),
                self.tracks[id2].momentum(after),
                self.state['Val1'][7 * id1:7 * id1 + 7, 7 * id2:7 * id2 + 7],
                self.fixMass[id1], self.fixMass[id2])\
            if after else super().correlation(id1, id2, after)

    def calculateNDF(self):
        self.ndf = 1

    def makeCoreMatrix(self):
        al1_prime = self.state['al1'].copy()
        al1_sum = np.zeros((6,1)) if self.fitVertex else np.zeros((4,1))
        size = len(self.tracks) * 7

        charges = speedOfLight * magneticField * np.array([trk.charge for trk in self.tracks])

        energy, idx = [], 0
        for trk, fixed, ch in zip(self.tracks, self.fixMass, charges):
            # Rotations in magnetic field in the xy plane
            if self.fitVertex:
                al1_prime[:2] += ch * np.array([[0, -1], [1, 0]]) @ (al1_prime[size:size+2] - al1_prime[idx+4:idx+6])
            elif self.atDecayPoint:
                al1_prime[:2] += ch * np.array([[0, -1], [1, 0]]) @ (self.before['vertex'][:2] - al1_prime[idx+4:idx+6])

            e = np.sqrt(trk.mass(False)**2 + np.sum(al1_prime[idx:idx + 3]**2))
            energy.append(e)

            al1_sum[3] += e if fixed else al1_prime[idx + 3]
            al1_sum[:3] += al1_prime[idx:idx + 3]
            if self.fitVertex and fixed:
                assert e != 0
                al1_sum[-2:] += ch * al1_prime[idx:idx+2:-1] / e
            idx += 7

        self.state['d'] = np.array(massSqFromP4(al1_sum[:4].ravel()) - self.targetMass**2).reshape(-1, 1)
        self.state['D'] = np.empty(self.state['al0'].shape)

        ilo, ihi = 0, 7
        for trk, fixed, ch, e in zip(self.tracks, self.fixMass, charges, energy):
            self.state['D'][ilo:ihi] = self.stateD(al1_sum, al1_prime[ilo:ilo + 3], ch, e, fixed)
            ilo, ihi = ihi, ihi + 7

        if self.fitVertex:
            self.state['D'][-1] = 0
            self.state['D'][-3:-1] = -2 * np.array([
                [-al1_sum[4], al1_sum[1]],
                [ al1_sum[5], al1_sum[0]]
            ]) @ np.array([al1_sum[3], charges.sum()])
    
    def stateD(self, al1_sum, al1_cur, ch, e, fixed):
        d45 = np.zeros((2, 1))
        if self.fitVertex or not self.atDecayPoint:
            d45 = al1_sum[:2] if fixed else al1_sum[:2] - al1_sum[3] * al1_cur[:2]

        d0123 = np.concatenate([
            al1_sum[3] * al1_cur / e - al1_sum[:3],
            np.array([0]).reshape(-1, 1)
        ]) if fixed else al1_sum[:4]

        return 2 * np.concatenate([
            np.array([1, 1, 1, -1]).reshape(-1, 1) * d0123,
            ch * np.array([[ 0, 1], [-1, 0]]) @ d45 / e,
            np.array([0]).reshape(-1, 1)
        ])

    def __checkFixMass(self):
        if not len(self.fixMass):
            self.fixMass = [False] * len(self.tracks)
        else:
            assert len(self.fixMass) == len(self.tracks)
    
    def updateParent(self, particle):
        kmp = MakeParent()
        kmp.tracks = self.tracks
        kmp.correlation = self.correlations
        kmp.vertex = self.after['vertex']
        if self.fitVertex:
            kmp.vertexError = self.after['vtxErr']
            kmp.trackVtxError = self.after['trkVtxErr']
        
        kmp.doMake()

        particle.extra.update({
            'chisq' : self.chisq,
              'ndf' : self.ndf,
             'prob' : 0.,
        })
        particle.momentum = kmp.parent.momentum(False)
        particle.vertex = kmp.parent.position(False)
        particle.errmtx = kmp.parent.errmtx(False)
