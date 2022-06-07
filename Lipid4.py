from Lipid import Lipid

# For backwards-compatibility with older simulation scripts
class Lipid4(Lipid):
    def __init__(self, system, midPos=[0.,0.,0.], theta=0., phi=0., lipidType={"Head": 0, "Mid1": 1, "Mid2": 1, "Tail": 1}, k_bend=10.):
        super().__init__(system, midPos, theta, phi, lipidType, k_bend)