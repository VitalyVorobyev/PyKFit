import sys
sys.path.append('../lib')

import numpy as np

from helix import helixParams, helixJacobian

def test_jacobian():
    rng = np.random.default_rng(seed=0)
    pos = rng.uniform(-1, 1, 3)
    mom = rng.uniform(-1, 1, 3)
    q, B = 1, 1.5
    eps = 1.e-6

    h = helixParams(pos, mom, q, B)
    jac = h.
