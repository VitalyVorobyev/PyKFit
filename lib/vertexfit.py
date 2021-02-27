""" """

from fitbase import FitBase
from particle import Particle

import numpy as np
from scipy.linalg import block_diag

class VertexFit(FitBase):
    def __init__(self, v0, v0err, tracks=[]):
        super().__init__(2, 10)
        self.tracks = []
        self.trksize = 5  # helix parameters
        self.descendants_updated = False
    
    