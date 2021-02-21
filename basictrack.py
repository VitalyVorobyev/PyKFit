""" """

import numpy as np

class BasicTrack:
    def __init__(self, p4, p4err):
        self.fourMomentum = p4
        self.fourMomentumError = p4err

    @property
    def mass(self):
        return 
