""" """

import numpy as np
import abc

from fittrack import KFitTrack
from errormaker import make_error_type1_pair

def check_zero_energy(mom):
    assert mom[-1] != 0

def check_invertable(mtx):
    assert np.linalg.det(mtx) != 0

def inverse_similarity(A, B):
    sim = A @ B @ A.T()
    check_invertable(sim)
    return np.linalg.inv(sim)

class FitBase(abc.ABC):
    """ Abstract base class for kinematic fitters """
    def __init__(self):
        self.tracks = []
        self.correlations = []
        self.fitted = False
        self.state = {}
        self.necessaryTrackCount = 999
        self.max_iterations = 10

    def addTrack(self, p, x, e, q):
        self.tracks.append(KFitTrack.makeTrackBefore(p, x, e, q))

    def trackChisq(self, idx):
        assert self.fitted and idx >= 0 and idx < len(self.tracks)
        return self.tracks[idx].chisqPars

    def correlation(self, id1, id2, after):
        if after:
            assert self.fitted
            return make_error_type1_pair(
                self.tracks[id1].momentum(True),
                self.tracks[id2].momentum(True),
                self.state['Val1'][6*id1:6*id1+6, 6*id2:6*id2+6])
        
        if id1 == id2:
            return self.tracks[id1].before['errmtx']

        idx1, idx2 = sorted([id1, id2])
        # WTF?
        index = len(self.tracks) * idx1 - idx1 * (idx1 - 1) / 2 + 2*idx2 - idx1 - 1
        return self.correlation[index] if id1 == idx1 else self.correlation[index].T()

    def prepareInputMatrix(self):
        raise NotImplementedError

    def calculateNDF(self):
        raise NotImplementedError

    def makeCoreMatrix(self):
        raise NotImplementedError

    def doFit1(self):
        assert len(self.tracks) >= self.necessaryTrackCount

        self.prepareInputMatrix()
        self.calculateNDF()

        chisq, chisq_tmp = 0., 1.e10
        self.state['ala'] = self.state['al0']
        alp_tmp = {key: self.state[key] for key in ['al1', 'Val1', 'ala']}

        for i in range(self.max_iterations):
            self.makeCoreMatrix()

            
