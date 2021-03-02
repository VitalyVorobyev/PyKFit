#! /usr/bin/env python
""" """

import numpy as np

from lib.vertexfit import VertexFit
from lib.vertexdecaysample import d0pipi
from lib.particle import Particle

def d0pipiFit():
    tracks = d0pipi(sigma=0.003)

    d0before = Particle(
        charge=sum([trk.charge for trk in tracks]),
        momentum=sum([trk.momentum for trk in tracks]),
        errmtx=None
    )
    print(f'{d0before.mass:.9f}')

    vtx = np.zeros(3)
    vtxErr = np.eye(3) * 100

    vfit = VertexFit(vtx, vtxErr, tracks)
    vfit.doFit()
    d0after = vfit.makeParent()
    
    print(f'{d0after.mass:.9f} {d0after.extra["chisq"]:.4f}')
    print(d0after.position)
    print(d0after.momentum)
    print(d0after.errmtx)

def main():
    d0pipiFit()

if __name__ == '__main__':
    main()
