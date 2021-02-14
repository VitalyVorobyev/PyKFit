""" """

import numpy as np

def make_beta(p):
    return p[:3].reshape(-1, 1) / p[3]

def init_7x7_from_6x6(err):
    hsm = np.empty((7, 7))
    hsm[:3, :3] = err[:3, :3]
    hsm[4:, 4:] = err[3:, 3:]
    hsm[:3, 4:] = err[:3, 3:]
    hsm[4:, :3] = err[3:, :3]
    return hsm

def fill_energy_single(hm, p, err_pp, err_px):
    assert p[-1] != 0
    beta = make_beta(p)
    hm[3, :3] = hm[:3, 3] = beta @ err_pp
    hm[3, 4:] = hm[4:, 3] = beta @ err_px
    hm[3, 3] = beta @ err_pp @ beta.T()
    return hm

def fill_energy_pair(hm, p1, p2, err_pp, err_px):
    assert p1[-1] != 0 and p2[-1] != 0
    beta1, beta2 = [make_beta(item) for item in [p1, p2]]
    hm[3, :3] = err_pp @ beta1.T()
    hm[3, 4:] = err_px @ beta1.T()
    hm[:3, 3] = beta2 @ err_pp
    hm[4:, 3] = beta2 @ err_px.T()
    hm[3, 3] = beta2 @ err_pp @ beta1.T()
    return hm

def make_error_type1_single(p, err):
    """      p: [px, py, pz, energy]
           err: 6x6 error matrix
        output: 7x7 error matrix
    """
    return fill_energy_single(
        init_7x7_from_6x6(err), p, err[:3, :3], err[:3, -3:])

def make_error_type1_pair(p1, p2, err):
    return fill_energy_pair(
        init_7x7_from_6x6(err), p1, p2, err[:3, :3], err[:3, 3:])

def make_error_vertex_track(p, err):
    """ Error between vertex and track
           err: 3x6
        output: 3x7
    """
    return np.column_stack([err[:,:3], make_beta(p) @ err[:,:3], err[:,-3:]])

def make_error_type3_single(p, err, fixed_mass):
    """      p: [px, py, pz, energy]
           err: 7x7 error matrix
        output: 7x7 error matrix
    """
    return err.copy() if fixed_mass else\
        fill_energy_single(err.copy(), p, err[:3,:3], err[:3,-3:])

def make_error_type3_pair(p1, p2, err, fixed_mass1, fixed_mass2):
    """      p: [px, py, pz, energy]
           err: 7x7 error matrix
        output: 7x7 error matrix
    """
    hm = err.copy()
    if fixed_mass1 and fixed_mass2:
        return fill_energy_pair(hm, p1, p2, err[:3,:3], err[:3,-3:])
    if fixed_mass1 and not fixed_mass2:
        hm[3,:] = make_beta(p1) @ err[:3, :]
    elif not fixed_mass1 and fixed_mass2:
        hm[:, 3] = err[:, :3] @ make_beta(p2).T()
    return hm

def make_error_type4(p, err):
    """      p: [px, py, pz, energy]
           err: 3x7 error matrix
        output: 3x7 error matrix
    """
    return np.column_stack([err[:,:3], make_beta(p) @ err[:,:3], err[:,-3:]])
