""" """

import numpy as np
import abc

from .fittrack import KFitTrack
from .errormaker import make_error_type1_pair

def check_zero_energy(mom):
    assert mom[-1] != 0

def check_invertable(mtx):
    assert np.linalg.det(mtx) != 0

def inverse_similarity(A, B) -> float:
    """ A: [NxN], B: [Nxd] -> [dxd] """
    sim = B.T @ A @ B
    check_invertable(sim)
    return np.linalg.inv(sim)

class FitBase(abc.ABC):
    """ Abstract base class for kinematic fitters """
    def __init__(self, ntracks, niter):
        self.tracks = []
        self.correlations = []
        self.fitted = False
        self.state = {}
        self.necessaryTrackCount = ntracks
        self.max_iterations = niter

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
        return self.correlation[index] if id1 == idx1 else self.correlation[index].T

    def fillInputMatrix(self):
        raise NotImplementedError

    def updateDescendants(self):
        raise NotImplementedError

    def ndf(self):
        raise NotImplementedError

    def calculateGradients(self):
        raise NotImplementedError

    def doFit(self):
        assert len(self.tracks) >= self.necessaryTrackCount
        self.fillInputMatrix()
    
        chisq, chisq_tmp = 0., 1.e10
        self.state.update({
            'ala' : self.state['al0'].copy(),
            'al1' : self.state['al0'].copy(),
            'Val1': np.empty(self.state['Val0'].shape),
        })
        alp_tmp = {key: self.state[key] for key in ['al1', 'Val1', 'ala']}

        for i in range(self.max_iterations):
            self.calculateGradients()
            chisq = self.__in_loop_calculation()
            print(f'iteration {i}/{self.max_iterations} {chisq:.6f} {chisq_tmp:.6f}')

            if chisq_tmp < chisq + 1.e-6:
                chisq = chisq_tmp
                for key, value in alp_tmp.items():
                    self.state[key] = value
                break
            else:
                chisq_tmp = chisq
                for key in alp_tmp.keys():
                    alp_tmp[key] = self.state[key]

                if i == self.max_iterations - 1:
                    self.state['ala'] = alp_tmp['al1']
                    self.maxIterationReached = True

        self.updateDescendants()
        self.chisq = chisq
        self.fitted = True

    def __in_loop_calculation(self):
        self.state['VD'] = self.__updated_VD()
        offset = self.__offset()
        self.state['lam'] = self.state['VD'] @ offset
        chisq = self.state['lam'].T @ offset
        self.state['al1'] = self.__updated_al1()
        self.state['Val1'] = self.__updated_Val1()

        return chisq.item()

    def __updated_VD(self):
        return inverse_similarity(self.state['Val0'], self.state['D'])

    def __offset(self):
        """ [dx1] + [Nxd].T x ([Nx1] - [Nx1]) -> [dx1] """
        return self.state['d'] + self.state['D'].T @ (self.state['al0'] - self.state['al1'])

    def __updated_al1(self):
        """ [Nx1] - [NxN] x [Nxd] x [dx1] """
        return self.state['al0'] - self.state['Val0'] @ self.state['D'] * self.state['lam']

    def __updated_Val1(self):
        """ [NxN] - [NxN] x [Nxd] x [dxd] x [Nxd].T x [NxN] -> [NxN] """
        return self.state['Val0'] - self.state['Val0'] @ self.state['D'] @ self.state['VD'] @ self.state['D'].T @ self.state['Val0']
