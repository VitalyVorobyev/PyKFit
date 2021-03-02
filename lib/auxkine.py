""" """

import numpy as np

def energyFromPMass(p, mass):
    return np.sqrt(mass**2 + np.sum(p**2))

def momentumFromMass(p3, mass):
    return np.array(list(p3) + [energyFromPMass(p3, mass)])

def make_beta(p):
    return p[:3].reshape(-1, 1) / p[3]

def massSqFromP3E(p3, e):
    return e**2 - np.sum(p3**2)

def massSqFromP4(p4):
    return massSqFromP3E(p4[:3], p4[3])

def massFromP4(p4):
    return np.sqrt(massSqFromP3E(p4[:3], p4[3]))

def kallen(x, y, z):
    return (x + y + z) * (x - y - z) * (x - y + z) * (x + y - z)

def twoBodyMomentum(mp, m1, m2=None):
    """ Parent -> """
    if m2 is None:
        m2 = m1
    return 0.5 * np.sqrt(kallen(mp, m1, m2)) / mp

