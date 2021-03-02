#! /usr/bin/env python
""" """

import numpy as np

from lib.basicmassfit import BasicMassFit
from lib.decaysample import d0pipi
from lib.particle import Particle

def d0pipiFit():
    tracks = d0pipi(sigma=0.003)

    d0before = Particle(
        charge=sum([trk.charge for trk in tracks]),
        momentum=sum([trk.momentum for trk in tracks]),
        errmtx=None
    )
    print(f'{d0before.mass:.9f}')
    
    mfit = BasicMassFit(1.865)
    mfit.tracks = tracks
    mfit.fixMass = np.ones(len(mfit.tracks), dtype=bool)
    mfit.doFit()

    d0after = mfit.makeParent()
    
    print(f'{d0after.mass:.9f} {d0after.extra["chisq"]:.4f}')

def main():
    d0pipiFit()

if __name__ == '__main__':
    main()
