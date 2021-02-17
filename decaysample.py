""" """

import numpy as np
from scipy.spatial.transform import Rotation

from fittrack import KFitTrack

MD0 = 1.865
MPI = 0.145

def get_mass(p):
    pflat = p.ravel()
    return np.sqrt(pflat[-1]**2 + np.sum(pflat[:-1]**2))

def d0pipi(sigma=0.001, seed=None):
    """ D0 -> pi+ pi- """
    rng = np.random.default_rng(seed=seed)

    p0 = np.array([0, 0, np.sqrt(0.25*MD0**2 - MPI**2)])

    alpha, beta, gamma = rng.uniform(-np.pi, np.pi, 3)
    rot = Rotation.from_euler('zyx', [alpha, beta, gamma]).as_matrix()
    errors = sigma * rng.standard_normal(6)

    p1 = rot @  p0 + errors[:3]
    p2 = rot @ -p0 + errors[-3:]

    e1, e2 = [np.sqrt(MPI**2 + np.sum(p**2)) for p in [p1, p2]]
    errmtx = np.diag(sigma**2 * np.ones(7))

    p1 = np.array(list(p1) + [e1])
    p2 = np.array(list(p2) + [e2])

    return (
        KFitTrack.makeTrackBefore(p1, np.zeros(3), errmtx, +1),
        KFitTrack.makeTrackBefore(p2, np.zeros(3), errmtx, -1)
    )

def main():
    """ Test """
    import matplotlib.pyplot as plt

    N = 10**3
    tracks = [d0pipi() for _ in range(N)]

    dmass = [get_mass(t1.momentum(False) + t2.momentum(False)) for t1, t2 in tracks]
    print(dmass[:3])

    plt.hist(dmass, bins=60, histtype='step')
    plt.show()

if __name__ == '__main__':
    main()
