import numpy as np
import scipy.special as sp
from scipy.optimize import root

from Lipid import Lipid

#############################################################
#
# Geometry Setup Functions
#
#############################################################

# Note: this will attempt to make a cylindrical membrane with the requested
#       radius. However, the lipid placement scheme is very rudimentary, and
#       the actual equilibrium radius of the cylinder will almost always be
#       different from the requested r.
def cylindricalBilayer(system, r, typeOut, typeIn=None, z0=2.0, ao=1.163, ai=None, verbose=False):
    if ai == None:
        ai = ao
    if typeIn == None:
        typeIn = typeOut
    
    lipids = []

    #initial lipid spacing (linear dimension)
    do = np.sqrt(ao)
    di = np.sqrt(ai)
    
    #radius to inner and outer leaflet
    ri = r - z0
    ro = r + z0
    
    #number if lipids to place in inner and outer leaflets
    ni = int(2.*np.pi*ri/di)
    no = int(2.*np.pi*ro/do)
    
    #angular spacing for inner and outer lipids
    dthetai = 2.*np.pi/ni
    dthetao = 2.*np.pi/no
    
    Lz = system.box_l[2]
    
    if verbose: print("Placing Inner Leaflet")
    for z in np.linspace(0,Lz-di,int(Lz/di)):
        #place the inner ring of lipids
        for m in range(ni):
            pos = np.array([(system.box_l[0]/2.)+(ri*np.cos(m*dthetai)),(system.box_l[1]/2.)+(ri*np.sin(m*dthetai)),z])
            lipids.append(Lipid(system, midPos=pos, theta=np.pi/2., phi=np.pi+(m*dthetai), lipidType=typeIn))
    
    if verbose: print("Placing Outer Leaflet")
    for z in np.linspace(0,Lz-do,int(Lz/do)):
        #place outer ring of lipids
        for m in range(no):
            pos = np.array([(system.box_l[0]/2.)+(ro*np.cos(m*dthetao)),(system.box_l[1]/2.)+(ro*np.sin(m*dthetao)),z])
            lipids.append(Lipid(system, midPos=pos, theta=np.pi/2., phi=m*dthetao, lipidType=typeOut))
    
    return lipids


# Creates bilayer buckled in x-direction of (approximately) total length L.
# arc-len increments ds1, ds2 allow the user to specify different lipid spacings
# for each leaflet, allowing for the creation of number-asymmetric membranes
def buckledBilayer(system, L, type1, type2=None, ds1=1.1, ds2=1.1, z0=2.0, verbose=False):
    if type2 == None:
        type2 = type1
    #truncated power series approximation for m, see Hu et al. 2013 JCP
    def M(g):
        return g - (0.125*np.power(g,2)) - (0.03125*np.power(g,3)) - (0.0107421875*np.power(g,4))
    
    def x(s, lam, m):
        return (2*lam*sp.ellipeinc(sp.ellipj(s/lam,m)[3],m))-s
    
    def z(s, lam, m):
        return 2*lam*np.sqrt(m)*(1.0-sp.ellipj(s/lam,m)[1])
    
    def psi(s, lam, m):
        return 2*np.arcsin(np.sqrt(m)*sp.ellipj(s/lam,m)[0])
    
    lipids = []
    
    dy1 = ds1
    dy2 = ds2
    Lx = system.box_l[0]
    Ly = system.box_l[1]
    Lz = system.box_l[2]
    
    zOffset = Lz/4.
    
    #see paper for explanation of parameters
    m = M((L-Lx)/L)
    lam = L/(4*sp.ellipk(m))
    
    Y1 = np.linspace(0,Ly-dy1,int(np.floor(Ly/dy1)))
    Y2 = np.linspace(0,Ly-dy2,int(np.floor(Ly/dy2)))
    
    S1 = np.linspace(0,L-ds1,int(np.floor(L/ds1)))
    S2 = np.linspace(0,L-ds2,int(np.floor(L/ds2)))
    
    if verbose: print("Placing Upper Leaflet")
    #Upper Leaflet
    for Y in Y1:
        #Stepping along outer arc length
        for S in S1:
            #Figure out how far along midplane arc length we are
            s = root(lambda t: t - S - z0*psi(t,lam,m), S).x[0]
            PSI = psi(s, lam, m)
            
            X = x(s, lam, m) - z0*np.sin(PSI)
            Z = z(s, lam, m) + z0*np.cos(PSI) + zOffset
            
            lipids.append(Lipid(system, midPos=np.array([X,Y,Z]), theta=-PSI, lipidType=type1))
    
    n = len(lipids)
    if verbose:
        print("Number of lipids in upper leaflet: {}".format(n))
        print("Placing Lower Leaflet")
    
    #Lower Leaflet
    z0 = -z0
    for Y in Y2:
        #Stepping along outer arc length
        for S in S2:
            #Figure out how far along midplane arc length we are
            s = root(lambda t: t - S - z0*psi(t,lam,m), S).x[0]
            PSI = psi(s, lam, m)
            
            X = x(s, lam, m) - z0*np.sin(PSI)
            Z = z(s, lam, m) + z0*np.cos(PSI) + zOffset
            
            lipids.append(Lipid(system, midPos=np.array([X,Y,Z]), theta=np.pi-PSI, lipidType=type2))
    
    if verbose: print("Number of lipids in lower leaflet: {}".format(len(lipids)-n))
    
    return lipids


def flatBilayer(system, numA, typeA, numB=None, typeB=None, verbose=False, z0=2.0):
    if(numB is None):
        numB = numA
    
    if(typeB is None):
        typeB = typeA
    
    Lx  = system.box_l[0]
    Ly  = system.box_l[1]
    Lz  = system.box_l[2]
    
    sites = []
    
    for n in [numA, numB]:
        nx = np.sqrt(Lx*n/Ly)
        ny = np.sqrt(Ly*n/Lx)
        
        if(np.floor(nx)*np.ceil(ny) >= n) and (np.floor(nx)*np.ceil(ny) <= np.ceil(nx)*np.floor(ny)):
            nx = np.floor(nx)
            ny = np.ceil(ny)
        elif(np.ceil(nx)*np.floor(ny) >= n) and (np.ceil(nx)*np.floor(ny) <= np.floor(nx)*np.ceil(ny)):
            nx = np.ceil(nx)
            ny = np.floor(ny)
        else:
            nx = np.ceil(nx)
            ny = np.ceil(ny)
        
        xgrid = np.linspace(0,Lx,num=int(nx),endpoint=False)
        ygrid = np.linspace(0,Ly,num=int(ny),endpoint=False)
        s = [(x,y) for x in xgrid for y in ygrid]
        
        dn = int(nx*ny - n)
        for i in range(dn):
            s.pop(int(np.random.rand()*len(s)))
        
        sites.append(s)
    
    lipids = []
    
    if verbose: print("Placing {} lipids in top leaflet".format(len(sites[0])))
    for site in sites[0]:
        pos = [site[0], site[1], (Lz/2.) + z0]
        lipids.append(Lipid(system, midPos=pos, lipidType=typeA))
    
    if verbose: print("Placing {} lipids in bottom leaflet".format(len(sites[1])))
    for site in sites[1]:
        pos = [site[0], site[1], (Lz/2.) - z0]
        lipids.append(Lipid(system, midPos=pos, theta=np.pi, lipidType=typeB))
    
    return lipids


#############################################################
#
# Analysis
#
#############################################################

def stray(system, lipid):
    nbhood = system.analysis.nbhood(lipid.getPos("Head"), r_catch = 3.0)
    if(len(nbhood) > 5):
        return False
    else:
        return True


# Analyze the radius of the cylindrical lipid structure oriented
# along the z-axis (assuming homogeneous)
def cylinderRadius(system, lipids, verbose = False):
    # First, calculate (x,y) location of Center Of Mass
    com = np.array([0.,0.,0.])
    num_mol = 0
    
    for l in lipids:
        if(not stray(system, l)):
            com += l.getPos()
            num_mol += 1
    
    com = com[:2]               # get rid of z component
    com = com / float(num_mol)  # average to get (x,y) CoM
    
    num_inner = 0
    num_outer = 0
    r_inner = 0.
    r_outer = 0.
    
    for l in lipids:
        if(not stray(system, l)):
            # tail to head vector, only keeping (x,y)
            tth = np.array(l.getPos("Head") - l.getPos("Tail") )[:2]
            #com to lipid vector
            r = l.getPos()[:2] - com
            # dot r with tth to determine leaflet orientation
            if(np.dot(tth,r) > 0):
                r_outer += np.linalg.norm(r)
                num_outer += 1
            else:
                r_inner += np.linalg.norm(r)
                num_inner += 1
    
    r_inner = r_inner / float(num_inner)
    r_outer = r_outer / float(num_outer)
    
    R = (r_outer + r_inner)/2.
    
    if verbose: print("Inner:{}   Outer:{}   Middle:{}".format(r_inner, r_outer, R))
    return R


# Determine in which leaflet lipid species are located
# Assumes flat membrane with surface normal in z direction
def leafletContent(system, lipids, typeA, typeB):
    topA   = 0
    topB   = 0
    botA   = 0
    botB   = 0
    nStray = 0

    unitZ = np.array([0,0,1])

    for l in lipids:
        if(not stray(system, l)):
            tth = np.array(l.getPos("Head") - l.getPos("Tail") )
            x = np.dot(tth, unitZ)

            if x < 0: # it is in the bottom
                if l.type == typeA:
                    botA += 1
                elif l.type == typeB:
                    botB += 1

            if x > 0: # it is in the top
                if l.type == typeA:
                    topA += 1
                elif l.type == typeB:
                    topB += 1
        else:
            nStray += 1

    return (topA, botA, topB, botB, nStray)
