""" """

import numpy as np

def alpha(B):
    return B

class Helix:
    """ Helix representation as in W. Hulsbergen NIM 552 (2005) 566  """
    def __init__(self, d0, phi0, omega, z0, tanl, errmtx=None):
        self.d0 = d0
        self.phi0 = phi0
        self.omega = omega
        self.z0 = z0
        self.tanl = tanl
        self.errmtx = errmtx

    @property
    def pars(self):
        return np.array([self.d0, self.phi0, self.omega, self.z0, self.tanl])

    def pt(self, q, B):
        """ Transverce momentum given particle charge and magnetic field
        Args:
            - q: particle charge [units of the positron charge]
            - B: magnetic field [T]
        """
        return q * alpha(B) / self.omega

    @property
    def rho(self):
        """ Radius of curvature """
        return 1. / self.omega

    def phi(self, length):
        """ length: flight length """
        return self.phi0 + self.omega * length

    def position(self, length):
        r, phi = self.rho, self.phi(length)
        return np.array([
             r * np.sin(phi) - (r + self.d0) * np.sin(self.phi0),
            -r * np.cos(phi) + (r + self.d0) * np.cos(self.phi0),
            self.z0 + length * self.tanl])
    
    def momentum(self, length, q, B):
        phi = self.phi(length)
        return self.pt(q, B) * np.array([np.cos(phi), np.sin(phi), self.tanl])

    def jacobian(self, length, q, B):
        """ d (r, p) / d (helix): [5x6] matrix """
        phi = self.phi(length)
        sphi, cphi = np.sin(phi), np.cos(phi)
        sphi0, cphi0 = np.sin(self.phi0), np.cos(self.phi0)
        r = self.rho
        lom, rsq = length * self.omega, r**2
        qalph = q * alpha(B)
        return np.array([
            [-sphi0, cphi0, 0, 0, 0, 0],
            [
                r * cphi - (r + self.d0) * cphi0,
                r * sphi - (r + self.d0) * sphi0, 0,
                -qalph * r * sphi, qalph * r * cphi, 0],
            [
                rsq * ( sphi0 - sphi + lom * cphi),
                rsq * (-cphi0 + cphi + lom * sphi), 0,
                -rsq * qalph * (cphi + lom * sphi),
                -rsq * qalph * (sphi - lom * cphi),
                -rsq * qalph * self.tanl],
            [0, 0, 1, 0, 0, 0],
            [0, 0, length, 0, 0, qalph * r],
        ])

    def cartesianCovariance(self, length, q, B):
        """ [6x6] covariariance matrix for Cartesian coordinates """
        if self.errmtx is None:
            return None
        
        jac = self.jacobian(length, q, B)
        return jac.T @ self.errmtx @ jac


def helixParams(pos, mom, q, B):
    x, y, z = pos
    px, py, pz = mom
    qalph = q * alpha(B)

    pt = np.hypot(px, py)
    phi = np.arctan2(py, px)

    px0 = px + y * qalph
    py0 = py - x * qalph

    pt0 = np.hypot(px0, py0)
    phi0 = np.arctan2(py0, px0)
    tanl = pz / pt

    # The flight length in the transverse plane, measured 
    # from the point of the helix closeset to the z-axis
    length = (phi - phi0) * pt / qalph
    return (
        Helix((pt0 - pt) / qalph, phi0, qalph / pt, z - length * tanl, tanl),
        length
    )

def helixJacobian(pos, mom, q, B):
    """ d (helix) / d (r, p): [6x5] matrix """
    x, y, _ = pos
    px, py, pz = mom
    qalph = q * alpha(B)

    px0 = px + y * qalph
    py0 = py - x * qalph

    pt = np.hypot(px, py)
    pt0 = np.hypot(px0, py0)
    pt0sq, ptsq, ptcu = pt0**2, pt**2, pt**3

    phi = np.arctan2(py, px)
    phi0 = np.arctan2(py0, px0)
    length = (phi - phi0) * pt / qalph

    return np.array([
        [-py0 / pt0, -qalph * px0 / pt0sq, 0, -pz * px0 / pt0sq, 0],
        [ px0 / pt0, -qalph * py0 / pt0sq, 0, -pz * py0 / pt0sq, 0],
        [0, 0, 0, 1, 0],
        [
            (px0 / pt0 - px / pt) / qalph,
            -py0 / pt0sq,
            -qalph * px / ptcu,
            -pz * (py0 / pt0sq - py / ptsq) / qalph,
            -pz * px / ptcu
        ],
        [
            (py0 / pt0 - py / pt) / qalph,
            px0 / pt0sq,
            -qalph * py / ptcu,
            pz * (px0 / pt0sq - px / ptsq) / qalph,
            -pz * py / ptcu
        ],
        [0, 0, 0, -length / pt, 1 / pt]
    ])

def makeHelix(pos, mom, q, B, errmtx=None):
    jac = helixJacobian(pos, mom, q, B)
    helix, length = helixParams(pos, mom, q, B)
    helix.errmtx = None if errmtx is None else jac.T @ errmtx @ jac
    return helix, length
