import sys
sys.path.append('lib')

import numpy as np
from helix import helixParams, helixJacobian, Helix, makeHelix
from helix import vertexOffset, vertexOffsetGradient

def test_helixJacobian():
    """ Numeric check of d (helix) / d (r, p) """
    rng = np.random.default_rng(seed=0)
    pos = rng.uniform(-1, 1, 3)
    mom = rng.uniform(-1, 1, 3)
    bfield, eps = 1.5, 1.e-6

    for q in [-1, 1]:
        h, _ = helixParams(pos, mom, q, bfield)
        jac = helixJacobian(pos, mom, q, bfield)

        assert jac.shape == (6, 5)

        for i in range(3):
            postmp = pos.copy()
            postmp[i] += eps
            htmp, _ = helixParams(postmp, mom, q, bfield)
            dhOverEps = (htmp.pars - h.pars) / eps

            assert np.allclose(dhOverEps, jac[i,:])

        for i in range(3):
            momtmp = mom.copy()
            momtmp[i] += eps
            htmp, _ = helixParams(pos, momtmp, q, bfield)
            dhOverEps = (htmp.pars - h.pars) / eps

            assert np.allclose(dhOverEps, jac[i + 3,:])

def test_jacobian():
    """ Numeric check of d (r, p) / d (helix) """
    rng = np.random.default_rng(seed=0)
    hpars = rng.uniform(-1, 1, 5)
    h = Helix(*hpars)
    length = rng.uniform(-1, 1)
    bfield, eps = 1.5, 1.e-6

    pos = h.position(length)
    for q in [-1, 1]:
        mom = h.momentum(length, q, bfield)
        jac = h.jacobian(length, q, bfield)

        assert jac.shape == (5, 6)

        for i in range(5):
            print(i)
            hparstmp = hpars.copy()
            hparstmp[i] += eps
            htmp = Helix(*hparstmp)

            dpos = (htmp.position(length) - pos) / eps
            dmom = (htmp.momentum(length, q, bfield) - mom) / eps

            assert np.allclose(dpos, jac[i,: 3])
            assert np.allclose(dmom, jac[i,-3:])

def test_cart_helix_cart():
    rng = np.random.default_rng(seed=0)
    pos = rng.uniform(-1, 1, 3)
    mom = rng.uniform(-1, 1, 3)
    bfield = 1.5

    for q in [-1, 1]:
        h, length = helixParams(pos, mom, q, bfield)

        assert np.allclose(pos, h.position(length))
        assert np.allclose(mom, h.momentum(length, q, bfield))

def test_helix_cart_helix():
    rng = np.random.default_rng(seed=0)
    hpars = rng.uniform(-1, 1, 5)
    length0 = rng.uniform(-1, 1)
    bfield = 1.5

    for q in [-1, 1]:
        hpars[2] = np.abs(hpars[2]) * q
        h0 = Helix(*hpars)
        h, length = helixParams(h0.position(length0), h0.momentum(length0, q, bfield), q, bfield)
        assert np.allclose([length0,], [length,])
        assert np.allclose(h0.pars, h.pars)

def test_invJacobians():
    """ Check if product of inverse jacobians is a unit matrix """
    rng = np.random.default_rng(seed=0)
    pos = rng.uniform(-1, 1, 3)
    mom = rng.uniform(-1, 1, 3)
    bfield = 1.5

    for q in [-1, 1]:
        h, length = helixParams(pos, mom, q, bfield)
        jac = helixJacobian(pos, mom, q, bfield)  # Jh = d (helix) / d (r, p)
        jacInv = h.jacobian(length, q, bfield)    # Jrp = d (r, p) / d (helix)

        assert jac.shape == (6, 5)
        assert jacInv.shape == (5, 6)
        assert np.allclose(jacInv @ jac, np.eye(5))
        assert np.allclose(jac @ jacInv, np.eye(6))

def test_cartesianCovariance():
    rng = np.random.default_rng(seed=0)
    pos = rng.uniform(-1, 1, 3)
    mom = rng.uniform(-1, 1, 3)
    dummy = 0.3 * rng.uniform(-1, 1, (6, 6))
    errmtx = np.eye(6) + dummy + dummy.T
    bfield = 1.5

    for q in [-1, 1]:
        h, length = makeHelix(pos, mom, q, bfield, errmtx)
        assert np.allclose(errmtx, h.cartesianCovariance(length, q, bfield))

def test_helixCovariance():
    rng = np.random.default_rng(seed=0)
    dummy = 0.3 * rng.uniform(-1, 1, (5, 5))
    errmtx = np.eye(5)  + dummy + dummy.T
    hpars = rng.uniform(-1, 1, 5)
    length0 = rng.uniform(-1, 1)

    bfield = 1.5

    for q in [-1, 1]:
        hpars[2] = np.abs(hpars[2]) * q
        h0 = Helix(*hpars, errmtx)
        h, length = makeHelix(
            h0.position(length0),
            h0.momentum(length0, q, bfield),
            q, bfield,
            h0.cartesianCovariance(length0, q, bfield)
        )

        assert np.allclose([length0,], [length,])
        assert np.allclose(h0.pars, h.pars)
        assert np.allclose(h0.errmtx, h.errmtx)

def test_vertexOffsetGradient():
    rng = np.random.default_rng(seed=0)
    hpars = rng.uniform(-1, 1, 5)
    vtx = rng.uniform(-1, 1, 3)
    eps = 1.e-6

    offset = vertexOffset(hpars, vtx)
    hparsGrad, vtxGrad = vertexOffsetGradient(hpars, vtx)

    assert offset.shape == (1, 2)
    assert hparsGrad.shape == (2, 5)
    assert vtxGrad.shape == (2, 3)

    for i in range(5):
        hparstmp = hpars.copy()
        hparstmp[i] += eps

        doffset = (vertexOffset(hparstmp, vtx) - offset) / eps
        assert np.allclose(hparsGrad[:, i], doffset)

    for i in range(3):
        vtxtmp = vtx.copy()
        vtxtmp[i] += eps

        doffset = (vertexOffset(hpars, vtxtmp) - offset) / eps
        assert np.allclose(vtxGrad[:, i], doffset)
