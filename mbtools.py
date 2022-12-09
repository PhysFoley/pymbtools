import numpy as np
import espressomd
import scipy.special as sp
from scipy.optimize import root

from Lipid import Lipid

#############################################################
#
# Geometry Setup Functions
#
#############################################################


def flatBilayer(system, nTop, nBot=None, z0=2.0, verbose=False):
    """
    Create a flat periodic bilayer in the xy-plane, centered
    in the middle of the box in the z-direction.
    
    Required Arguments:
    system -- the espresso system
    nTop   -- number of lipids in upper leaflet
    
    Keyword Arguments:
    nBot    -- number of lipids to place in lower leaflet. defaults to nTop
    z0      -- distance from bilayer midsurface to lipid midpoints
    verbose -- boolean for extra text output
    """
    
    if(nBot is None):
        nBot = nTop
    
    Lx  = system.box_l[0]
    Ly  = system.box_l[1]
    Lz  = system.box_l[2]
    
    sites = rectGrid(Lx, Ly, nTop, nBot)
    
    uAngle = np.array([[0.,0.]]*len(sites[0]))
    dAngle = np.array([[np.pi,0.]]*len(sites[0]))
    
    uPos = []
    dPos = []
    
    if verbose: print("Generating {} lipid sites in top leaflet".format(len(sites[0])))
    for site in sites[0]:
        uPos.append( [site[0], site[1], (Lz/2.) + z0] )
    
    if verbose: print("Generating {} lipid sites in bottom leaflet".format(len(sites[1])))
    for site in sites[1]:
        dPos.append( [site[0], site[1], (Lz/2.) - z0] )
    
    return np.array(uPos), np.array(dPos), uAngle, dAngle

def cylindricalBilayer(system, r, numOut, numIn, z0o=2.0, z0i=2.0, verbose=False):
    """
    Create a cylindrical tether bilayer periodic in the z-direction. Note: there is
    no guarantee the tether will keep the radius initially specified.
    
    Required Arguments:
    system -- the espresso system
    r      -- tether radius
    numOut -- number of lipids to place in outer leaflet
    numIn  -- number of lipids to place in inner leaflet
    
    Keyword Arguments:
    z0o     -- distance from bilayer midsurface to outer leaflet lipid midpoints
    z0i     -- distance from bilayer midsurface to inner leaflet lipid midpoints
    verbose -- boolean for extra text output
    """
    
    Lx  = system.box_l[0]
    Ly  = system.box_l[1]
    Lz  = system.box_l[2]
    
    # first, create a sites on a rectangular grid
    # the first coord in this grid wraps around the cylinder
    # the second coord is along the axial (z) direction
    sites = rectGrid(2.0*np.pi*r, Lz, numOut, numIn)
    
    outPos = []
    inPos  = []
    
    outAngle = []
    inAngle  = []
    
    if verbose: print(f"Calculating Outer Leaflet Configuration for {numOut} Lipids")
    for site in sites[0]:
        s = site[0]
        z = site[1]
        phi = s/r
        x = (r+z0o)*np.cos(phi)
        y = (r+z0o)*np.sin(phi)
        outPos.append( [(Lx/2.0)+x, (Ly/2.0)+y, z] )
        outAngle.append( [np.pi/2.0, phi] )
    
    if verbose: print(f"Calculating Inner Leaflet Configuration {numIn} Lipids")
    for site in sites[1]:
        s = site[0]
        z = site[1]
        phi = s/r
        x = (r-z0i)*np.cos(phi)
        y = (r-z0i)*np.sin(phi)
        inPos.append( [(Lx/2.0)+x, (Ly/2.0)+y, z] )
        inAngle.append( [np.pi/2.0, np.pi+phi] )
    
    return np.array(outPos), np.array(inPos), np.array(outAngle), np.array(inAngle)


def buckledBilayer(system, L, nTop, nBot=None, z0=2.0, verbose=False):
    """
    Create a bilayer buckled in the x-direction, with (approximate)
    total length L.
    
    Required Arguments:
    system -- the espresso system
    L      -- membrane length along buckle
    nTop   -- number of lipids to place in upper leaflet
    
    Keyword Arguments:
    nBot    -- number of lipids to place in lower leaflet. defaults to nTop
    z0      -- distance from bilayer midsurface to lipid midpoints
    verbose -- boolean for extra text output
    """
    
    if nBot == None:
        nBot = nTop
    
    #truncated power series approximation for m, see Hu et al. 2013 JCP
    def M(g):
        return g - (0.125*np.power(g,2)) - (0.03125*np.power(g,3)) - (0.0107421875*np.power(g,4))
    
    def x(s, lam, m):
        return (2*lam*sp.ellipeinc(sp.ellipj(s/lam,m)[3],m))-s
    
    def z(s, lam, m):
        return 2*lam*np.sqrt(m)*(1.0-sp.ellipj(s/lam,m)[1])
    
    def psi(s, lam, m):
        return 2*np.arcsin(np.sqrt(m)*sp.ellipj(s/lam,m)[0])
    
    Lx = system.box_l[0]
    Ly = system.box_l[1]
    Lz = system.box_l[2]
    
    zOffset = Lz/4.
    
    #see paper for explanation of parameters
    m = M((L-Lx)/L)
    lam = L/(4*sp.ellipk(m))
    
    sites = rectGrid(L, Ly, nTop, nBot)
    
    uPos = []
    dPos = []
    uAngle = []
    dAngle = []
    
    if verbose: print("Generating Upper Leaflet Geometry")
    #Upper Leaflet
    for (S,Y) in sites[0]:
        #Figure out how far along midplane arc length we are
        s = root(lambda t: t - S - z0*psi(t,lam,m), S).x[0]
        PSI = psi(s, lam, m)
        
        X = x(s, lam, m) - z0*np.sin(PSI)
        Z = z(s, lam, m) + z0*np.cos(PSI) + zOffset
        
        uPos.append(np.array([X,Y,Z]))
        uAngle.append([-PSI,0.])
    
    if verbose:
        print("Number of lipid sites in upper leaflet: {}".format(len(uPos)))
        print("Generating Lower Leaflet Geometry")
    
    #Lower Leaflet
    z0 = -z0
    for (S,Y) in sites[1]:
        #Figure out how far along midplane arc length we are
        s = root(lambda t: t - S - z0*psi(t,lam,m), S).x[0]
        PSI = psi(s, lam, m)
        
        X = x(s, lam, m) - z0*np.sin(PSI)
        Z = z(s, lam, m) + z0*np.cos(PSI) + zOffset
        
        dPos.append(np.array([X,Y,Z]))
        dAngle.append([np.pi-PSI,0.])
    
    if verbose: print("Number of lipid sites in lower leaflet: {}".format(len(dPos)))
    
    return np.array(uPos), np.array(dPos), np.array(uAngle), np.array(dAngle)


def rectGrid(Lx, Ly, numA, numB):
    """
    Generate rectangular grid of sites for the two leaflets.
    
    This is primarily a helper routine for the other geometry
    setup functions.
    """
    
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
    
    return sites


def placeLipids(ulipids, dlipids, uPos, dPos, uAngle, dAngle):
    """Place lipids into generated sites at specified angles."""
    
    if (len(ulipids) != len(uPos)) or (len(ulipids) != len(uAngle)):
        print("Mismatch between number of lipids and number of sites")
    if (len(dlipids) != len(dPos)) or (len(dlipids) != len(dAngle)):
        print("Mismatch between number of lipids and number of sites")
    
    for i in range(len(ulipids)):
        ulipids[i].displace(uPos[i])
        ulipids[i].rotate([0,1,0], uAngle[i][0])
        ulipids[i].rotate([0,0,1], uAngle[i][1])
    
    for i in range(len(dlipids)):
        dlipids[i].displace(dPos[i])
        dlipids[i].rotate([0,1,0], dAngle[i][0])
        dlipids[i].rotate([0,0,1], dAngle[i][1])


def springBondStructure(system,particles,spring_k,cutoff):
    """
    Bond together a collection of particles with Harmonic
    bonds of spring constant spring_k, where every particle
    is bonded to all neighbors within the cutoff distance.
    NOTE: This method is intended to be used on an initial
    configuration of particles that are within cutoff
    distance of one another in *absolute coordinates*, the
    minimum image distance is not used in this routine
    
    Required Arguments:
    system    -- the espresso system
    particles -- list of espresso particle handles to be bonded together
    spring_k  -- harmonic bond spring constant
    cutoff    -- neighbor-bonding cutoff distance
    """
    
    for p in particles:
        # find neighbors within cutoff dist
        # NOTE: this assumes the particles are actually close
        #       (i.e., does not calculate minimum image dist)
        #       (this should be true of initial config)
        nblist = []
        for p2 in particles:
            if (p != p2) and (np.linalg.norm(p.pos-p2.pos) < cutoff):
                nblist.append(p2)
        
        for nbp in nblist:
            # get ids of particles neighbor already bonded to
            nb_bonded_to = [b[1] for b in nbp.bonds]
            
            # if we're already bonded, go on to next neighbor
            if p.id in nb_bonded_to:
                pass
            else:
                r = np.linalg.norm(p.pos-nbp.pos)
                bond = espressomd.interactions.HarmonicBond(k=spring_k, r_0=r)
                system.bonded_inter.add(bond)
                p.add_bond((bond,nbp.id))

#############################################################
#
# Analysis
#
#############################################################

def stray(system, lipid):
    """
    Determine if a lipid has strayed from the bilayer.
    All this function does is check whether there are
    more than 5 beads within 3 sigma of the lipid's
    headbead, which is a totally arbitrary metric, but
    is good enough in most situations.
    """
    
    nbhood = system.analysis.nbhood(lipid.getPos("Head"), r_catch = 3.0)
    if(len(nbhood) > 5):
        return False
    else:
        return True


def cylinderRadius(system, lipids, verbose = False):
    """
    Calculate the average radius of a cylindrical tether
    oriented parallel to the z direction.
    """
    
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
            # tail-to-head vector, only keeping (x,y)
            tth = np.array(l.getPos("Head") - l.getPos("Tail") )[:2]
            #CoM to lipid midpoint vector
            r = l.getPos()[:2] - com
            # dot r with tth to determine which leaflet based on orientation
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


def leafletContent(system, lipids, typeA, typeB):
    """
    Determine in which leaflet lipid species are located.
    Assumes flat membrane with surface normal in z direction.
    Leaflet of residence is determined purely from lipid
    orientation relative to z direction, so the numbers
    reported should only be regarded as approximate.
    
    Required Arguments:
    system -- espresso system
    lipids -- list of lipid handles
    typeA  -- type dictionary for type A lipids
    typeB  -- type dictionary for type B lipids
    
    Returns a tuple containing:
    topA   -- number of A-lipids in upper leaflet
    botA   -- number of A-lipids in lower leaflet
    topB   -- number of B-lipids in upper leaflet
    botB   -- number of B-lipids in lower leaflet
    nStray -- number of stray lipids
    """
    
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
