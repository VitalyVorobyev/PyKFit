""" """

from auxkine import massFromP4

class Particle:
    def __init__(self, charge, momentum, errmtx, vertex=None, extra={}):
        self.charge = charge
        self.momentum = momentum
        self.errmtx = errmtx
        self.vertex = vertex
        self.extra = extra

    @property
    def mass(self):
        return massFromP4(self.momentum).item()

    @property
    def threeMomentumError(self):
        return self.errmtx[:3,:3]

    @property
    def threeMomentum(self):
        return self.momentum[:3]
