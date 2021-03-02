""" """

from .auxkine import massFromP4, momentumFromMass
from .helix import makeHelix

class Particle:
    def __init__(self, **kwargs):
        self.charge   = kwargs.get('charge', None)
        self.momentum = kwargs.get('momentum', None)
        self.errmtx   = kwargs.get('errmtx', None)
        self.position = kwargs.get('position', None)
        self.helix    = kwargs.get('helix', None)
        self.extra    = kwargs.get('extra', {})
        self.mass0    = kwargs.get('mass', None)

    @property
    def mass(self):
        if self.mass0 is not None:
            return self.mass0
        return massFromP4(self.momentum).item()

    @property
    def threeMomentumError(self):
        return self.errmtx[:3,:3]

    @property
    def threeMomentum(self):
        return self.momentum[:3]

    def buildHelix(self, bfield):
        self.helix, self.length = makeHelix(self.position, self.momentum, self.charge, bfield, self.errmtx)

    def buildCartesian(self, length, bfield=1.5):
        self.position = self.helix.position(length)
        self.momentum = momentumFromMass(
            self.helix.momentum(length, self.charge, bfield),
            self.mass
        )
        self.errmtx = self.helix.cartesianCovariance(length, self.charge, bfield)
