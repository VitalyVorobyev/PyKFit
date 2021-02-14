""" """

import numpy as np

def cutLine(mtx, idx):
    omtx = np.empty(mtx.shape[0] - 1, mtx.shape[1] - 1)
    omtx[:idx, :idx] = mtx[:idx  , :idx]
    omtx[idx:, idx:] = mtx[idx+1:, idx+1:]
    omtx[idx:, :idx] = mtx[idx+1:, :idx]
    omtx[:idx, idx:] = mtx[:idx  , idx+1:]
    return omtx

def cutElement(vec, idx):
    return np.concatenate([vec[:idx], vec[idx+1:]]).reshape(-1,1)

class KFitTrack:
    def __init__(self, **kwargs):
        self.before =  {
            'mompos' : np.concatenate([
                kwargs.get('pBefore', np.empty(4)),
                kwargs.get('xBefore', np.empty(3))
            ]).reshape(-1,1),
            'errmtx' : kwargs.get('eBefore', np.empty(7, 7))
        }
        self.after = {
            'mompos' : np.concatenate([
                kwargs.get('pAfter', np.empty(4)),
                kwargs.get('xAfter', np.empty(3))
            ]).reshape(-1,1),
            'errmtx' : kwargs.get('eAfter', np.empty(7, 7))
        }
        self.charge = kwargs['charge']
        self.vertex, self.vertex_error = None, None

    def momentum(self, after):
        return self.after['mompos'][:4] if after else self.before['mompos'][:4]

    def fitPars(self, afterFit):
        return cutElement(
            self.after['mompos'] if afterFit else self.before['mompos'], 3)

    def fitParsError(self, afterFit):
        """ TODO: consider caching this calculation """
        return cutLine(self.after['errmtx'] if afterFit else self.before['errmtx'], 3)

    @staticmethod
    def makeTrackBefore(p, x, e, q):
        return KFitTrack(pBefore=p, xBefore=x, eBefore=e, charge=q)

    @staticmethod
    def makeTrackAfter(p, x, e, q):
        return KFitTrack(pAfter=p, xAfter=x, eAfter=e, charge=q)

    @property
    def deltaFitPars(self):
        return self.fitPars(False) - self.fitPars(True)
    
    @property
    def deltaMomPos(self):
        return self.before['mompos'] - self.after['mompos']

    @property
    def chisqPars(self):
        delta = self.deltaFitPars
        return delta.T() @ np.linalg.inv(self.fitParsError(False)) @ delta

    @property
    def chisqMomPos(self):
        delta = self.deltaMomPos
        return delta.T() @ np.linalg.inv(self.before['errmtx']) @ delta
