import numpy as np
import scipy.special as sp

from Lipid4 import Lipid4

#############################################################
#
# Geometry Setup Functions
#
#############################################################

# Assumes a cubic box
def openEdgesBilayer(system, numLipids, type1, type2, verbose=False):
    lipids = []
    dx = 1.125
    dy = 1.125
    lpl = int(system.box_l[0]/dy) #lipids per line
    lines = int(numLipids/(lpl*2)) #divide numlipids by two because two leaflets
    
    #because of the division by two, this only works perfectly for even number of lipids atm
    for i in range(int(numLipids/2)):
        x = (system.box_l[0]/2.)-(lines*dx/2) + int(i/lpl)*dx
        y = (i%lpl)*dy
        
        if verbose: print("Placing lipid at {},{},{}".format(x,y,(system.box_l[0]/2.)+3.))
        lipids.append(Lipid(system, midPos=np.array([x,y,(system.box_l[0]/2.)+1.5]), theta=0, phi=0, lipidType=type1))
        lipids.append(Lipid(system, midPos=np.array([x,y,(system.box_l[0]/2.)-1.5]), theta=np.pi, phi=0, lipidType=type2))
    
    return lipids



def cylindricalBilayer(system, r, type1, type2):
    lipids = []

    #length of lipid chain and (largest) diameter of lipid beads (for spacing)
    l = 3.0
    d = 1.125
    
    #number if lipids in inner and outer leaflets (ni, no repspectively)
    ni = int(2.*np.pi*(r-l)/d)
    no = int(2.*np.pi*r/d)

    #radius of center bead for inner and outer lipids
    ri = r - (l/2.)
    ro = r + (l/2.)

    #angular spacing for inner and outer lipids
    dthetai = 2.*np.pi/ni
    dthetao = 2.*np.pi/no
    
    for z in np.arange(0,system.box_l[2],d):
        #place the inner ring of lipids
        for m in range(ni):
            pos = np.array([(system.box_l[0]/2.)+(ri*np.cos(m*dthetai)),(system.box_l[1]/2.)+(ri*np.sin(m*dthetai)),z])
            lipids.append(Lipid(system, midPos = pos, theta=np.pi/2., phi=np.pi+(m*dthetai), lipidType=type1))
        
        #place outer ring of lipids
        for m in range(no):
            pos = np.array([(system.box_l[0]/2.)+(ro*np.cos(m*dthetao)),(system.box_l[1]/2.)+(ro*np.sin(m*dthetao)),z])
            lipids.append(Lipid(system, midPos = pos, theta=np.pi/2., phi=m*dthetao, lipidType=type2))
    
    return lipids


# Creates bilayer buckled in x-direction of total length L.
# ds1, ds2 allow the user to specify different lipid spacings
# for each leaflet, allowing for the creation of asymmetric membranes
def buckledBilayer(system, L, type1, type2, ds1=1.125, ds2=1.125, verbose=False):
    #truncated power series approximation for m, see paper
    def M(g):
        return g - (0.125*np.power(g,2)) - (0.03125*np.power(g,3)) - (0.0107421875*np.power(g,4))
    
    def x(s, lam, m):
        return (2*lam*sp.ellipeinc(sp.ellipj(s/lam,m)[3],m))-s
    
    def z(s, lam, m):
        return 2*lam*np.sqrt(m)*(1.0-sp.ellipj(s/lam,m)[1])
    
    def psi(s, lam, m):
        return 2*np.arcsin(np.sqrt(m)*sp.ellipj(s/lam,m)[0])
    
    def bisect(s, ds, lam, m, zo):
        L = s + ds - 0.5
        R = s + ds + 0.5
        FM = 0.
            
        while(np.absolute(FM-ds) >= 0.001):
            FL = L - s - zo*(psi(L, lam, m)-psi(s, lam, m))
            FR = R - s - zo*(psi(R, lam, m)-psi(s, lam, m))
        
            if((FL-ds)*(FR-ds) > 0.):
                print("Bisection failed")
        
            Mid = (L + R)/2.
            FM = Mid - s - zo*(psi(Mid, lam, m)-psi(s, lam, m))
            if(FM > ds):
                R = Mid
            else:
                L = Mid
        
        return Mid
    
    lipids = []
    
    dy1 = ds1
    dy2 = ds2
    zo = 2.25
    Lx = system.box_l[0]
    Ly = system.box_l[1]
    Lz = system.box_l[2]
    
    zOffset = Lz/4.
    
    #see paper for explanation of parameters
    m = M((L-Lx)/L)
    lam = L/(4*sp.ellipk(m))
    
    Y1 = np.linspace(0,Ly-dy1,int(np.floor(Ly/dy1)))
    Y2 = np.linspace(0,Ly-dy2,int(np.floor(Ly/dy2)))
    
    if verbose: print("Placing Upper Leaflet")
    #Upper Leaflet
    for y in Y1:
        S = 0.
        X = 0.
        Z = zOffset+zo
        lipids.append(Lipid4(system, midPos=np.array([X,y,Z]), theta=0., phi=0., lipidType=type1))
        
        while(X < Lx):
            S = bisect(S, ds1, lam, m, zo)
            PSI = psi(S, lam, m)
            X = x(S, lam, m) - zo*np.sin(PSI)
            Z = z(S, lam, m) + zo*np.cos(PSI) + zOffset
            if(PSI >= 0.):
                p = np.pi
                t = PSI
            else:
                p = 0.
                t = np.abs(PSI)
            # added -ds/2 to stop it from placing a lipid overlapping with first lipid
            if(X < Lx - ds1/2):
                lipids.append(Lipid4(system, midPos=np.array([X,y,Z]), theta=t, phi=p, lipidType=type1))
    
    n = len(lipids)
    if verbose:
        print("Number of lipids in upper leaflet: {}".format(n))
        print("Placing Lower Leaflet")
    
    #Lower Leaflet
    zo = -zo
    for y in Y2:
        S = 0.
        X = 0.
        Z = zOffset+zo
        lipids.append(Lipid4(system, midPos=np.array([X,y,Z]), theta=np.pi, phi=0., lipidType=type2))
        
        while(X < Lx):
            S = bisect(S, ds2, lam, m, zo)
            PSI = psi(S, lam, m)
            X = x(S, lam, m) - zo*np.sin(PSI)
            Z = z(S, lam, m) + zo*np.cos(PSI) + zOffset
            if(PSI >= 0.):
                p = 0.
                t = PSI
            else:
                p = np.pi
                t = np.abs(PSI)
            
            if(X < Lx - ds2/2):
                lipids.append(Lipid4(system, midPos=np.array([X,y,Z]), theta=np.pi - t, phi=p, lipidType=type2))
    
    if verbose: print("Number of lipids in lower leaflet: {}".format(len(lipids)-n))
    
    return lipids


# Less stable than buckledBilayer
# Buckled in the x-direction, L will be total length
def naiveBuckledBilayer(system, L, type1, type2):
    #truncated power series approximation for m, see paper
    def M(g):
        return g - (0.125*np.power(g,2)) - (0.03125*np.power(g,3)) - (0.0107421875*np.power(g,4))
    
    def x(s, lam, m):
        return (2*lam*sp.ellipeinc(sp.ellipj(s/lam,m)[3],m))-s
    
    def z(s, lam, m):
        return 2*lam*np.sqrt(m)*(1.0-sp.ellipj(s/lam,m)[1])
    
    def psi(s, lam, m):
        return 2*np.arcsin(np.sqrt(m)*sp.ellipj(s/lam,m)[0])
    
    lipids = []
    
    ds = 1.125 #modify this value to properly deal with area per lipid
    dy = 1.125 #this also
    Lx = system.box_l[0]
    Ly = system.box_l[1]
    Lz = system.box_l[2]
    
    #number of lipids per monolayer line
    n = int(np.floor(L/ds))
    
    #see paper for explanation of parameters
    m = M((L-Lx)/L)
    lam = L/(4*sp.ellipk(m))
    
    #arc length values for each lipid, evenly spaced from 0 to L (not including the end, that overlaps!)
    S = np.linspace(0,L-ds,n)
    Y = np.linspace(0,Ly-dy,np.floor(Ly/dy))
    
    X = []
    Z = []
    PSI = []
    #calculate x and z positions, also
    #lipid tilt from z axis, t, is just psi
    #but need to be careful about negatives,
    #should only be positive with phi to match
    for s in S:
        X.append(x(s,lam,m))
        Z.append(z(s,lam,m) + Lz/2.)
        PSI.append(psi(s,lam,m))
    
    for y in Y:
        for i in range(n):
            if(PSI[i] >= 0.):
                p = np.pi
                pp = 0.
                t = PSI[i]
            else:
                p = 0.
                pp = np.pi
                t = np.abs(PSI[i])
            lipids.append(Lipid(system, midPos=np.array([X[i],y,Z[i]+1.5]), theta=t, phi=p, lipidType=type1))
            lipids.append(Lipid(system, midPos=np.array([X[i],y,Z[i]-1.5]), theta=np.pi-t, phi=pp, lipidType=type1))
    
    return lipids

def flatBilayer(system, numA, typeA, numB=None, typeB=None, verbose=False, zo=1.5, k=10.):
    # mid bead distance from midplane is zo
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
        pos = [site[0], site[1], (Lz/2.) + zo]
        lipids.append(Lipid4(system, midPos=pos, lipidType=typeA, k_bend=k))
    
    if verbose: print("Placing {} lipids in bottom leaflet".format(len(sites[1])))
    for site in sites[1]:
        pos = [site[0], site[1], (Lz/2.) - zo]
        lipids.append(Lipid4(system, midPos=pos, theta=np.pi, lipidType=typeB, k_bend=k))
    
    return lipids

# Assumes an approximately flat bilayer oriented with
# its normal in the z direction. Buckles it in the x direction.
# Takes as arguments the system, the array of lipids,
# and the new Lx box dimension.
def mapBuckle(system, lipids, Lx):
    #truncated power series approximation for m, see paper
    def M(g):
        return g - (0.125*np.power(g,2)) - (0.03125*np.power(g,3)) - (0.0107421875*np.power(g,4))
    
    def x(s, lam, m):
        return (2*lam*sp.ellipeinc(sp.ellipj(s/lam,m)[3],m))-s
    
    def z(s, lam, m):
        return 2*lam*np.sqrt(m)*(1.0-sp.ellipj(s/lam,m)[1])
    
    def psi(s, lam, m):
        return 2*np.arcsin(np.sqrt(m)*sp.ellipj(s/lam,m)[0])
    
    # Calculate approximate midplane z coord
    # Note: should really calculate separately
    #       for each bilayer, then average
    # NOTE: Currently unused
    zmid = 0.
    for l in lipids:
        zmid += l.getPos()
    zmid = zmid / len(lipids)
    
    L  = system.box_l[0]
    Ly = system.box_l[1]
    Lz = system.box_l[2]
    
    zOffset = Lz/4.
    zo = 2.
    
    #see paper for explanation of parameters
    m = M((L-Lx)/L)
    lam = L/(4*sp.ellipk(m))
    
    # separate top and bottom leaflets
    
    top = []
    bot = []
    unitZ = np.array([0,0,1])

    for l in lipids:
        if(not stray(system, l)):
            tth = np.array(l.getPos("Head") - l.getPos("Tail") )

            if np.dot(tth, unitZ) < 0: #it's in the bottom leaflet
                bot.append(l)
            else:
                top.append(l)
        else:
            pass #stray lipid
    
    #TODO: could use average distance between leaflets as 2*zo
    
    for l in top:
        # x position of midbead becomes displacement along buckle
        s = l.getPos()[0]
        
        # displacement vector
        d = np.array([x(s,lam,m) - s, 0., z(s,lam,m) + zOffset + zo - l.getPos()[2] ])
        l.displace(d)
        
        # amount by which to rotate (in xz plane)
        theta = psi(s,lam,m)
        
        # normal displacement away from bilayer midplane (to alleviate overlap)
        nd = np.abs(np.sin(theta)) * 0.5 * np.array([-np.sin(theta),0.,np.cos(theta)])
        l.displace(nd)
        
        if(theta > 0.):
            l.rotate([0,1,0], theta)
            l.rotate([0,0,1], np.pi)
        else:
            l.rotate([0,1,0], np.abs(theta))
            
    for l in bot:
        # x position of midbead becomes displacement along buckle
        s = l.getPos()[0]
        
        #displacement vector
        d = np.array([x(s,lam,m) - s, 0., z(s,lam,m) + zOffset - zo - l.getPos()[2] ])
        l.displace(d)
        
        # amount by which to rotate (in xz plane)
        theta = psi(s,lam,m)
        
        # normal displacement away from bilayer midplane (to alleviate overlap)
        nd = -np.abs(np.sin(theta)) * 0.5 * np.array([-np.sin(theta),0.,np.cos(theta)])
        l.displace(nd)
        
        if(theta > 0.):
            l.rotate([0,1,0], theta)
            l.rotate([0,0,1], np.pi)
        else:
            l.rotate([0,1,0], np.abs(theta))
    
    # resize the system box to the new buckle size
    system.box_l = [Lx, system.box_l[1], system.box_l[2]]
    
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
            #com to head vector
            r   = np.array(l.getPos("Head") )[:2] - com
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
