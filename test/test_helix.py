import sys
sys.path.append('lib')

import numpy as np
from helix import helixParams, helixJacobian, Helix

def test_helixJacobian():
    """ Numeric check of d (helix) / d (r, p) """
    rng = np.random.default_rng(seed=0)
    pos = rng.uniform(-1, 1, 3)
    mom = rng.uniform(-1, 1, 3)
    B, eps = 1.5, 1.e-6

    for q in [-1, 1]:
        h, _ = helixParams(pos, mom, q, B)
        jac = helixJacobian(pos, mom, q, B)

        assert jac.shape == (6, 5)

        for i in range(3):
            postmp = pos.copy()
            postmp[i] += eps
            htmp, _ = helixParams(postmp, mom, q, B)
            dhOverEps = (htmp.pars - h.pars) / eps

            assert np.allclose(dhOverEps, jac[i,:])

        for i in range(3):
            momtmp = mom.copy()
            momtmp[i] += eps
            htmp, _ = helixParams(pos, momtmp, q, B)
            dhOverEps = (htmp.pars - h.pars) / eps

            assert np.allclose(dhOverEps, jac[i + 3,:])

def test_jacobian():
    """ Numeric check of d (r, p) / d (helix) """
    rng = np.random.default_rng(seed=0)
    hpars = rng.uniform(-1, 1, 5)
    h = Helix(*hpars)
    length = rng.uniform(-1, 1)
    B, eps = 1.5, 1.e-6

    pos = h.position(length)
    for q in [-1, 1]:
        mom = h.momentum(length, q, B)
        jac = h.jacobian(length, q, B)

        assert jac.shape == (5, 6)

        for i in range(5):
            print(i)
            hparstmp = hpars.copy()
            hparstmp[i] += eps
            htmp = Helix(*hparstmp)

            dpos = (htmp.position(length) - pos) / eps
            dmom = (htmp.momentum(length, q, B) - mom) / eps

            assert np.allclose(dpos, jac[i,: 3])
            assert np.allclose(dmom, jac[i,-3:])
