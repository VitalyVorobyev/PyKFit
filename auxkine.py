""" """

import numpy as np

def momentumFromMass(p3, mass):
    return np.array(list(p3) + [np.sqrt(mass**2 + np.sum(p3**2))])

def make_beta(p):
    return p[:3].reshape(-1, 1) / p[3]

def massSqFromP3E(p3, e):
    return e**2 - np.sum(p3**2)

def massSqFromP4(p4):
    return massSqFromP3E(p4[:3], p4[3])

