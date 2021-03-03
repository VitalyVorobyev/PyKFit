""" """

import numpy as np
from scipy.spatial.transform import Rotation

from .particle import Particle
from .helix import Helix, makeHelix
from .auxkine import twoBodyMomentum

MD0 = 1.865
MPI = 0.145

def d0pipi(sigma=0.002, seed=None):
    """ D0 -> pi+ pi- """
    p0 = np.array([0, 0, twoBodyMomentum(MD0, MPI)])
    position = np.zeros(3)
    errmtx = np.eye(5) * sigma**2

    rng = np.random.default_rng(seed=seed)
    alpha, beta, gamma = rng.uniform(-np.pi, np.pi, 3)
    rot = Rotation.from_euler('zyx', [alpha, beta, gamma]).as_matrix()

    h01, _ = makeHelix(position, rot @  p0, +1, 1.5)
    h02, _ = makeHelix(position, rot @ -p0, -1, 1.5)

    # errors = np.zeros((2,5)) # sigma * rng.standard_normal((2, 5))
    errors = sigma * rng.standard_normal((2, 5))
    hpars1 = h01.pars + errors[0]
    hpars2 = h02.pars + errors[1]

    pion1 = Particle(charge=+1, mass=MPI, helix=Helix(*hpars1, errmtx))
    pion2 = Particle(charge=-1, mass=MPI, helix=Helix(*hpars2, errmtx))

    pion1.buildCartesian(0)
    pion2.buildCartesian(0)

    return (pion1, pion2)
