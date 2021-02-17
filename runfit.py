#! /usr/bin/env python
""" """

import numpy as np
from massfit import MassFit
from decaysample import d0pipi

class Particle:
    def __init__(self):
        self.extra = {}
        self.momentum = np.zeros(4)

    @property
    def mass(self):
        return np.sqrt(self.momentum.ravel()[-1]**2 - np.sum(self.momentum.ravel()[:3]**2))

def d0pipiFit():
    tracks = d0pipi()

    d0_before = Particle()
    d0_before.momentum = sum([trk.momentum(False) for trk in tracks])
    print(f'{d0_before.mass:.3f}')
    
    mfit = MassFit(1.865)
    mfit.tracks = tracks
    mfit.fixMass = np.ones(len(mfit.tracks), dtype=bool)
    mfit.prepareInputMatrix()
    mfit.doFit()

    d0 = Particle()
    mfit.updateParent(d0)

    print(f'{d0.mass:.3f}')

def main():
    d0pipiFit()

if __name__ == '__main__':
    main()
