#! /usr/bin/env python

""" """

import matplotlib.pyplot as plt
import numpy as np

from lib.vertexdecaysample import d0pipi
from lib.auxkine import massFromP4

def main():
    """ Test """

    N = 10**3
    tracks = [d0pipi(sigma=0.002) for _ in range(N)]

    ptot = np.array([massFromP4(t1.momentum + t2.momentum) for t1, t2 in tracks])
    print(ptot[:3])
    print(ptot.mean())
    print(ptot.std())

    plt.hist(ptot, bins=60, histtype='step')
    plt.show()

if __name__ == '__main__':
    main()
