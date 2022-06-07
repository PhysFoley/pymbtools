import numpy as np
from scipy.spatial.transform import Rotation
from espressomd.interactions import FeneBond, HarmonicBond

class Lipid:
    def __init__(self, system, midPos=[0.,0.,0.], theta=0., phi=0., lipidType={"Head": 0, "Mid": 1, "Tail": 1}, k_bend=10.):
        self.type = lipidType
        initRestLength = 0.95
        
        p = {}
        #Initially place lipid vertically, then rotate
        if len(lipidType) == 3:
            headPos = midPos + (initRestLength*np.array([0, 0, 1]))
            tailPos = midPos + (initRestLength*np.array([0, 0, -1]))
            p = {"Head": headPos, "Mid": midPos, "Tail": tailPos}
        elif len(lipidType) == 4:
            headPos = midPos + (initRestLength*np.array([0, 0, 1.5]))
            mid1Pos = midPos + (initRestLength*np.array([0, 0, .5]))
            mid2Pos = midPos + (initRestLength*np.array([0, 0, -.5]))
            tailPos = midPos + (initRestLength*np.array([0, 0, -1.5]))
            p = {"Head": headPos, "Mid1": mid1Pos, "Mid2": mid2Pos, "Tail": tailPos}
        else:
            print("Error: Only 3-bead and 4-bead lipids are supported")
            print("       Received lipidType with len {}".format(len(lipidType)))
        
        self.bead = {}
        for beadName in self.type:
            self.bead[beadName] = system.part.add(pos=p[beadName], type=lipidType[beadName])
        
        if(theta != 0.):
            self.rotate([0,1,0], theta)
        if(phi != 0.):
            self.rotate([0,0,1], phi)
        
        # Set up FENE bond
        k_bond = 30.
        d_r_max = 1.5
        fene = FeneBond(k=k_bond, d_r_max=d_r_max)
        system.bonded_inter.add(fene)
        
        # Set up Harmonic bond
        r_0 = 4.
        harmonicBond = HarmonicBond(k=k_bend, r_0=r_0)
        system.bonded_inter.add(harmonicBond)
        
        if len(lipidType) == 3:
            self.bead["Head"].add_bond((fene, self.bead["Mid"].id))
            self.bead["Mid"].add_bond((fene, self.bead["Tail"].id))
            self.bead["Head"].add_bond((harmonicBond, self.bead["Tail"].id))
        elif len(lipidType) == 4:
            self.bead["Head"].add_bond((fene, self.bead["Mid1"].id))
            self.bead["Mid1"].add_bond((fene, self.bead["Mid2"].id))
            self.bead["Mid2"].add_bond((fene, self.bead["Tail"].id))
            self.bead["Head"].add_bond((harmonicBond, self.bead["Mid2"].id))
            self.bead["Mid1"].add_bond((harmonicBond, self.bead["Tail"].id))
    
    # Rotate lipid about axis of unit vector u
    # passing through lipid CoM
    # by angle a, in radians
    def rotate(self, u, a):
        R = Rotation.from_rotvec(a*np.array(u))
        
        com = np.array([0.,0.,0.])
        for name in self.type:
            com += self.bead[name].pos
        com = com / float(len(self.type))
        
        for name in self.type:
            relpos = self.bead[name].pos - com
            newpos = R.apply(relpos)
            self.bead[name].pos = newpos + com
    
    # Displace lipid by vector dr, moving all beads
    # by the same amount and preserving orientation
    def displace(self, dr):
        for beadName in self.type:
            self.bead[beadName].pos = np.copy(self.bead[beadName].pos) + dr
    
    # Get lipid position, or a bead position
    def getPos(self, beadName = "CoM"):
        #If default requested, return CoM coords
        if(beadName == "CoM"):
            com = np.array([0.,0.,0.])
            for name in self.type:
                com += self.bead[name].pos
            return com / float(len(self.type))
        #otherwise, return position of requested bead
        else:
            return np.copy(self.bead[beadName].pos)
