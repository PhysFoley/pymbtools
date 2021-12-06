import numpy as np
from scipy.spatial.transform import Rotation
from espressomd.interactions import FeneBond, HarmonicBond

class Lipid4:
    
    # Takes in the position of the Middle Tail bead
    def __init__(self, system, midPos=[0.,0.,0.], theta=0., phi=0., lipidType={"Head": 0, "Mid1": 1, "Mid2": 1, "Tail": 1}, k_bend=10.):
        self.type = lipidType
        initRestLength = 0.95
        #Initially place lipid vertically, then rotate
        headPos = midPos + initRestLength * np.array([0, 0, 1.5])
        mid1Pos = midPos + initRestLength * np.array([0, 0, .5])
        mid2Pos = midPos + initRestLength * np.array([0, 0, -.5])
        tailPos = midPos + initRestLength * np.array([0, 0, -1.5])
        p = {"Head": headPos, "Mid1": mid1Pos, "Mid2": mid2Pos, "Tail": tailPos}

        self.bead = {}
        for beadName in self.type:
            self.bead[beadName] = system.part.add(pos=p[beadName], type=lipidType[beadName])
        
        if(theta != 0.):
            self.rotate([0,1,0], theta)
        if(phi != 0.):
            self.rotate([0,0,1], phi)
        
        self._setupInternalSprings(system, k_bend)

    # Rotate lipid about axis of unit vector u
    # passing through lipid CoM
    # by angle a, in radians
    def rotate(self, u, a):
        # Rotation matrix, for arbitary axis and angle
        R = Rotation.from_rotvec(a*np.asarray(u))
        
        com = np.array([0.,0.,0.])
        for name in self.type:
            com += self.bead[name].pos
        com = com / float(len(self.type))
        
        for name in self.type:
            relpos = self.bead[name].pos - com
            newpos = R.apply(relpos)
            self.bead[name].pos = com + newpos
    
    # Displace lipid by vector dr, moving all beads
    # by the same amount and preserving orientation
    def displace(self, dr):
        for beadName in self.type:
            self.bead[beadName].pos = np.copy(self.bead[beadName].pos) + dr
    
    def getPos(self, beadName = "CoM"):
        #If default requested, return CoM coords
        if(beadName == "CoM"):
            com = np.array([0.,0.,0.])
            for name in self.type:
                com += self.bead[name].pos
            return com / float(len(self.type))
        else:
            return np.copy(self.bead[beadName].pos)
    
    # coords is a dict of positions like p in the constructor above, this allows
    # to modify any/all of the bead positions in one call
    def setBeadPos(self, coords):
        for name in coords:
            self.bead[name].pos = np.copy(coords[name])
    
    # Sets up FENE Bond and Harmonic Bond
    def _setupInternalSprings(self, system, k_bend):
        # Set Up FENE
        k_bond = 30.
        d_r_max = 1.5
        fene = FeneBond(k=k_bond, d_r_max=d_r_max)

        system.bonded_inter.add(fene)
        self.bead["Head"].add_bond((fene, self.bead["Mid1"].id))
        self.bead["Mid1"].add_bond((fene, self.bead["Mid2"].id))
        self.bead["Mid2"].add_bond((fene, self.bead["Tail"].id))

        # Set up Harmonic Bond
        r_0 = 4.0
        harmonicBond = HarmonicBond(k=k_bend, r_0=r_0)

        system.bonded_inter.add(harmonicBond)
        self.bead["Head"].add_bond((harmonicBond, self.bead["Mid2"].id))
        self.bead["Mid1"].add_bond((harmonicBond, self.bead["Tail"].id))

