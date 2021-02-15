""" """

import numpy as np

from fittrack import KFitTrack
from config import speedOfLight, magneticField

class MakeParent:
    def __init__(self):
        self.tracks = []
        self.trackVtxError = []
        self.correlation = []
        self.vertex = np.zeros(3)
        self.vertexError = None
        self.after = True
        self.atDecayPoint = True

    def addTrack(self, p, x, e, q, after):
        self.tracks.append(KFitTrack.makeTrackAfter(p, x, e, q) if after else
                           KFitTrack.makeTrackBefore(p, x, e, q))
    
    def doMake(self):
        charge = sum([trk.charge for trk in self.tracks])

        dMdC = self.calculateDMdC()
        ec = self.calculateError()
        em = dMdC @ ec @ dMdC.T()

        momentum = sum([trk.momentum(self.after) for trk in self.tracks])
        if self.atDecayPoint:
            dxy = -speedOfLight * magneticField * np.array([
                trk.charge * (self.vertex[:2] - trk.position(self.after)[:2]) for trk in self.tracks])
            momentum[:2] += dxy[::-1] * [-1, 1]

        self.parent = KFitTrack.makeTrackBefore(momentum, self.vertex, em, charge)

    def calculateError(self):
        index = len(self.tracks) * 7
        err = np.zeros((index + 3, index + 3))

        ilo, ihi = 0, 7
        for trk in self.tracks:
            err[ilo:ihi, ilo:ihi] = trk.errmtx(self.after)
            ilo, ihi = ihi, ihi + 7

        if self.correlation:
            assert len(self.correlation) == len(self.tracks) * (len(self.tracks) - 1)
            idx, jdx = 0, 7
            for corr in self.correlation:
                err[idx:idx+7, jdx:jdx+7] = err[jdx:jdx+7, idx:idx+7] = corr
                if jdx == index - 7:
                    idx, jdx = idx + 7, 0
                else:
                    jdx += 7

        if self.vertexError:
            err[index:, index:] = self.vertexError

        if self.trackVtxError:
            assert len(self.trackVtxError) == len(self.tracks)
            jdx = 0
            for corr in self.trackVtxError:
                err[index:index+3, jdx:jdx+7] = corr
                err[jdx:jdx+7, index:index+3] = corr.T()
                jdx += 7

    def calculateDMdC(self):
        index = len(self.tracks) * 7
        sum_a, dMdC = 0, np.zeros((7, index + 3))
        avec = -speedOfLight * magneticField * np.array([trk.charge for trk in self.tracks])

        for idx, a in enumerate(avec):
            index = idx * 7
            dMdC[0, 5 + index] =  a
            dMdC[1, 4 + index] = -a
            dMdC[:4, index:index+4] = np.eye(4)
        
        index = len(self.tracks) * 7
        sum_a = avec.sum()
        dMdC[4:, index:] = np.eye(3)
        dMdC[0, index + 1] = -sum_a
        dMdC[1, index] = sum_a

        return dMdC
