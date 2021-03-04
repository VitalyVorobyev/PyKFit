#! /usr/bin/env python
""" """

import numpy as np
import matplotlib.pyplot as plt

from lib.vertexfit import VertexFit
from lib.vertexdecaysample import d0pipi
from lib.particle import Particle
from lib.helixplot import plot_helix

def d0pipiFit():
    tracks = d0pipi(sigma=0.003)

    figax = plot_helix(tracks[0].helix)
    figax = plot_helix(tracks[1].helix, figax=figax)

    d0before = Particle(
        charge=sum([trk.charge for trk in tracks]),
        momentum=sum([trk.momentum for trk in tracks]),
        errmtx=None
    )
    print(f'{d0before.mass:.4f}')

    vtx = np.zeros(3)
    vtxErr = np.eye(3) * 100

    vfit = VertexFit(vtx, vtxErr, tracks)
    vfit.doFit()
    d0after = vfit.makeParent()
    
    print(f'{d0after.mass:.4f} {d0after.extra["chisq"]:.4f}')

    vertex = vfit.vertex()
    print(f'vertex: {vertex}')

    lengths = [trk.helix.lengthAtZ(vertex[2]) for trk in vfit.tracks]
    positions = [trk.helix.position(leng) for leng, trk in zip(lengths, vfit.tracks)]

    figax = plot_helix(vfit.tracks[0].helix, figax=figax)
    figax = plot_helix(vfit.tracks[1].helix, figax=figax)
    figax[1][0].plot([vertex[0]], [vertex[1]], 'o', markersize=10)
    figax[1][0].plot([positions[0][0]], [positions[0][1]], 'o', markersize=10)
    figax[1][0].plot([positions[1][0]], [positions[1][1]], 'o', markersize=7)
    plt.show()

    print(f'lengths: {lengths}')
    print(f'positions: {positions}')

def main():
    d0pipiFit()

if __name__ == '__main__':
    main()
