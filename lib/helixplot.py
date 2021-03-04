""" """

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size' : 16})
import matplotlib.pyplot as plt

def plot_helix(h, zlo=-1, zhi=1, figax=None):
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6)) if figax is None else figax

    z = np.linspace(zlo, zhi, 1000)
    length = h.lengthAtZ(z)
    pos = h.position(length)
    pos0 = h.position(0)
    rho = np.hypot(pos[0, :], pos[1, :])

    ax[0].plot(pos[ 0, :], pos[1, :])
    ax[0].plot([pos0[0]], [pos0[1]], 'o')
    
    ax[1].plot(pos[-1, :], rho)
    ax[1].plot([pos0[-1]], [np.hypot(pos0[0], pos0[1])], 'o')

    if figax is not None:
        for a in ax:
            a.minorticks_on()
            a.grid(which='major')
            a.grid(which='minor', linestyle=':')
            a.axis('equal')
        fig.tight_layout()

    return (fig, ax)

