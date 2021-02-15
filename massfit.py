""" """

from fitbase import FitBase
from makeparent import MakeParent
from errormaker import make_error_type3_single
from config import speedOfLight, magneticField

import numpy as np
from scipy.linalg import block_diag

def momentumFromMass(p3, mass):
    return np.array(list(p3) + [np.sqrt(mass**2 + np.sum(p3**2))])

class MassFit(FitBase):
    def __init__(self):
        self.necessaryTrackCount = 2
        self.fixMass = []

        self.after = {
            'vertex' : np.zeros(3),
            'vtxErr' : None,
            'trkVtxErr' : [],
        }

        self.before = {
            'vertex' : np.zeros(3),
            'vtxErr' : None,
            'trkVtxErr' : [],
        }

        self.fitVertex = False
    
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

        self.state.update({'al0' : np.concatenate(pars), 'Val0': block_diag(errs)})

        if self.correlations:
            self.prepareCorrelation()

        self.state.update({
             'al1' : self.state['al0'],
            'Val1' : np.empty(self.state['Val0'].shape),
               'D' : np.empty(self.state['al0'].shape)
        })

    def prepareOutputMatrix(self):
        index = 0
        for trk, fixed in zip(self.tracks, self.fixMass):
            curpars = self.state['al1'][index:index+7]
            if fixed:
                curpars[:4] = momentumFromMass(curpars[:3], trk.mass(False))
            trk.before['mompos'] = curpars
            trk.before['errmtx'] = make_error_type3_single(
                curpars[:4], self.state['Val1'][index:index+7, index:index+7])
            index += 7

        if self.fitVertex:
            self.after['vertex'] = self.state['al1'][-3:]
            self.after['vtxErr'] = self.state['Val1'][-3:, -3:]
        else:
            self.after = self.before


    def calculateNDF(self):
        self.ndf = 1

    def makeCoreMatrix(self):
        if self.fitVertex:
            self.__makeCoreMatrixWVertex()
        else:
            self.__makeCoreMatrixWoVertex()
    
    def __makeCoreMatrixWVertex(self):
        al1_prime = self.state['al1']
        al1_sum = np.zeros(7)
        size = len(self.tracks) * 7

        charges = speedOfLight * magneticField * np.array([trk.charge for trk in self.tracks])
        al1_sum[6] = charges.sum()
        energy = []
        energyInv = []

        idx = 0
        for trk, fixed, ch in zip(self.tracks, self.fixMass, charges):
            al1_prime[idx    ] -= ch * (al1_prime[size + 1] - al1_prime[idx + 5])
            al1_prime[idx + 1] += ch * (al1_prime[size    ] - al1_prime[idx + 4])
            e = np.sqrt(trk.mass(False)**2 + np.sum(al1_prime[idx:idx + 3]**2))
            energy.append(e)

            if fixed:
                assert e != 0
                invE = 1. / e
                energyInv.append(invE)
                al1_sum[3:6] += np.array([e, al1_prime[idx + 1] * ch * invE, al1_prime[idx + 0] * ch * invE])
            else:
                al1_sum[3] += al1_prime[idx + 3]
            al1_sum[:3] += al1_prime[idx:idx + 3]
            idx += 7

        self.state['d'] = al1_sum[3]**2 - np.sum(al1_sum[:3]**2) - self.targetMass**2

        idx = 0
        for jdx, (trk, fixed, ch) in enumerate(zip(self.tracks, self.fixMass, charges)):
            if fixed:
                self.state['D'][idx:idx + 3] = 2 * al1_sum[3] * al1_prime[idx:idx + 3] * energyInv[jdx] - al1_sum[:3]
                self.state['D'][idx + 3] = 0
                self.state['D'][idx + 4:idx + 7] = np.array([
                    -2 * ch * (al1_sum[3] * al1_prime[idx + 1]) * energyInv[jdx] - al1_sum[1],
                     2 * ch * (al1_sum[3] * al1_prime[idx + 0]) * energyInv[jdx] - al1_sum[0], 0])
            else:
                self.state['D'][idx:idx + 3] = -2 * al1_sum[:3]
                self.state['D'][idx:idx + 4] =  2 * al1_sum[3]
                self.state['D'][idx + 4:idx + 7] = np.array([
                     2 * ch * al1_sum[1],
                    -2 * ch * al1_sum[0], 0])
            idx += 7

        self.state['D'][-3:] = np.array([
             2. * al1_sum[3] * al1_sum[4] - al1_sum[1] * al1_sum[6],
            -2. * al1_sum[3] * al1_sum[5] - al1_sum[0] * al1_sum[6], 0]]

    def __makeCoreMatrixWoVertex(self):
        al1_prime = self.state['al1']
        al1_sum = np.zeros(7)
        size = len(self.tracks) * 7
        charges = speedOfLight * magneticField * np.array([trk.charge for trk in self.tracks])
        al1_sum[6] = charges.sum()
        energy = []
        energyInv = []

        # TODO:

    def __checkFixMass(self):
        if not self.fixMass:
            self.fixMass = [False] * len(self.tracks)
        else:
            assert len(self.fixMass) == len(self.tracks)
    
    def updateParent(particle):
        pass
