from __future__ import print_function
import sys
import os
import time
import numpy as np

import espressomd
from espressomd.io.writer import vtf

from mbtools import *
from Lipid import Lipid

# Extra text output
verb = True

alpha_a = 0.0 # skew angle in degrees
alpha_b = 0.0

u_num = 512
d_num = 512
jobname = "easier_stickytape"

print(f"Skew Angle A = {alpha_a} degrees")
print(f"Skew Angle B = {alpha_b} degrees")

# System parameters
#############################################################
# Main simulation parameters:
skin          = 0.4
time_step     = 0.005
temperature       = 1.5
integrator_steps  = 200
sampling_interval = 10
iterations        = 1000

# Warmup parameters:
warm_tstep  = 0.0005
warm_steps  = 10
warm_n_time = 1000
min_dist    = 0.90
warm_cap    = 2.0

# New Interaction Parameters for Tapered Lipids
#############################################################
alpha_a = alpha_a * (np.pi/180.0) # convert to radians
alpha_b = alpha_b * (np.pi/180.0)

r = [0.0]*8
b = []

# A-type lipid bead radii
r[3] = 0.5 + 1.5*np.sin(alpha_a)
r[2] = 0.5 + 0.5*np.sin(alpha_a)
r[1] = 0.5 - 0.5*np.sin(alpha_a)
r[0] = 0.5 - 1.5*np.sin(alpha_a) - 0.025

# B-type lipid bead radii
r[7] = 0.5 + 1.5*np.sin(alpha_b)
r[6] = 0.5 + 0.5*np.sin(alpha_b)
r[5] = 0.5 - 0.5*np.sin(alpha_b)
r[4] = 0.5 - 1.5*np.sin(alpha_b) - 0.025

# b-param matrix (cross-bead "diameters")
for i in range(len(r)):
    b.append([]*len(r))
    for j in range(len(r)):
        b[i].append(r[i]+r[j])

if verb:
    # output for checking
    print(b)

# since all potential depths = 1, no need to
# go through lorentz-berthelot rule, sqrt(1*1)=1

lj_eps = 1.0
cshift = 1./4.
wc = 1.6

def main():
    # Set Up
    #############################################################
    system = espressomd.System(box_l = [40.0,16.0,60.0]) #periodic BC by default
    system.set_random_state_PRNG()
    system.time_step = warm_tstep
    system.cell_system.skin = skin
    
    # Membrane
    ############################################################
    typeOne = {"Head":0,"Mid1":1,"Mid2":2,"Tail":3}
    typeTwo = {"Head":4,"Mid1":5,"Mid2":6,"Tail":7}
    
    uPos, dPos, uAngle, dAngle = flatBilayer(system, nTop=u_num, nBot=d_num, verbose=verb)
    
    # Create lists of lipid handles of desired types
    ulipids = [Lipid(system,lipidType=typeOne) for i in range(u_num)]
    dlipids = [Lipid(system,lipidType=typeTwo) for i in range(d_num)]
    
    placeLipids(ulipids, dlipids, uPos, dPos, uAngle, dAngle)
    
    # combine lipid handles into single array
    lipids = ulipids + dlipids
    
    if verb:
        print("Total number of lipids: " + str(len(lipids)) )
        print("kT={}".format(temperature))
    
    system.box_l = [60.0,16.0,60.0] # expand the box to get open membrane edges
    
    # create tickytapes; this function is defined at the bottom of this file
    tape = makeStickyTapes(system)
    
    # Non bonded Interactions between the lipid beads
    #############################################################

    for i in range(len(b)):
        # note: j < i
        for j in range(i+1):
            if (i in [0,4]) or (j in [0,4]):
                # purely repulsive WCA potential
                system.non_bonded_inter[i, j].lennard_jones.set_params(
                                    epsilon=lj_eps, sigma=b[i][j],
                                    cutoff=np.power(2.0, 1.0/6.0)*b[i][j], shift="auto")
            elif (i,j) in [(5,1),(5,2),(6,1),(6,2)]:
                # flip-fix repulsive cross-leaflet midbead interaction
                system.non_bonded_inter[i, j].lennard_jones.set_params(
                                    epsilon=lj_eps, sigma=b[i][j],
                                    cutoff=np.power(2.0, 1.0/6.0)*b[i][j], shift="auto")
            else:
                system.non_bonded_inter[i, j].lennard_jones_cos2.set_params(
                                    epsilon=lj_eps, sigma=b[i][j],
                                    width=wc, offset=0.)
    
    #############################################################
    #  Output Files                                             #
    #############################################################
    try:
        os.mkdir(jobname)
    except OSError:
        print("Directory {} already exists or could not be created.".format(jobname))
    
    with open(jobname+"/trajectory.vtf", "w") as vtf_fp, open(jobname+"/energy.txt","w") as en_fp, \
        open(jobname+"/flipflop.txt","w") as ff_fp, open(jobname+"/ptensor.txt","w") as pt_fp:
        
        # write structure block as header
        vtf.writevsf(system, vtf_fp)
        # write initial positions as coordinate block
        vtf.writevcf(system, vtf_fp)
        #NOTE: WRITING INITIAL CONFIG FIRST
        
        #############################################################
        #  Warmup Integration                                       #
        #############################################################
        
        if verb:
            print("Starting warmup integration:")
            print("Forces capped at {} epsilon/sigma".format(warm_cap) )
            print("No more than {} iterations of {} steps of length {} tau".format(warm_n_time, warm_steps, warm_tstep) )
            print("Proceeding to main integration if minimum distance > {} sigma".format(min_dist) )
        
        i = 0
        #global warm_cap
        system.force_cap = warm_cap
        act_min_dist = system.analysis.min_dist()
        while i < warm_n_time and act_min_dist < min_dist:
            system.integrator.run(warm_steps)
            act_min_dist = system.analysis.min_dist()
            if verb: print("Warmup iteration {}, current min_dist = {}".format(i, act_min_dist))
            i += 1
        
        system.force_cap = 0.0 # Disable force cap
        system.time_step = time_step # Switch to main integration time step
        
        if verb:
            print("\nWarm up finished\n")
            print("box_l is Lx={}, Ly={}, Lz={}\n".format(*system.box_l) )
        
        #############################################################
        #  Thermostat                                               #
        #############################################################
        
        ##
        ## To simulate the NPT-ensemble, un-comment the following two lines
        ## and comment out the Langevin thermostat a few lines down
        ##
        #system.thermostat.set_npt(kT=temperature, gamma0=1.0, gammav=0.0002)
        #system.integrator.set_isotropic_npt(ext_pressure=0.0, piston=0.01, direction=[1,1,0])
        
        ##
        ## Langevin thermostat for simulating NVT-ensemble
        ##
        system.thermostat.set_langevin(kT=temperature, gamma=1.0, seed=int(np.random.rand()*10000))
        
        #############################################################
        #  Main Simulation                                          #
        #############################################################
        start_time = time.time()
        
        # Each iteration of the outermost loop advances the
        # simulation time by integrator_steps*time_step tau
        for i in range(1, iterations+1):
            system.integrator.run(integrator_steps)
            
            if(i % sampling_interval == 0):
                vtf.writevcf(system, vtf_fp)
                vtf_fp.write("unitcell {} {} {}\n".format(*np.copy(system.box_l)) )
                
                e = system.analysis.energy()
                en_fp.write("{},{},{},{}\n".format(e["total"],e["kinetic"],e["bonded"],e["non_bonded"]))
                
                pt = system.analysis.stress_tensor()
                pt_fp.write("{},{},{},{},{},{}\n".format(pt["total"][0][0],pt["total"][1][1],pt["total"][2][2],pt["total"][1][0],pt["total"][2][0],pt["total"][2][1]))
                
                f = leafletContent(system, lipids, typeOne, typeTwo)
                ff_fp.write("{},{},{},{},{}\n".format(*f))
                
                en_fp.flush()
                pt_fp.flush()
                vtf_fp.flush()
                ff_fp.flush()
                print("Completed {} iterations (tau={})".format(i,i*integrator_steps*time_step))
    
    elapsed = time.time() - start_time
    print(jobname + " Done. Elapsed time {} seconds".format(elapsed))


# Sticky Tape Setup Function
############################################################
def makeStickyTapes(system):
    typeS1 = 8 # sticky for type one lipids
    typeS2 = 9 # sticky for type two lipids
    typeSN = 10 # not sticky, just WCA
    
    ygrid = np.linspace(0.0,system.box_l[1]-1.0,num=int(system.box_l[1]))
    zgrid = np.linspace(-3.0,3.0,num=6)
    
    tape_parts = [] # array for storing particle handles
    
    # make the sticky beads
    for y in ygrid:
        for z in zgrid:
            if z > 0:
                tape_parts.append(system.part.add(pos=[-1.0,y,(system.box_l[2]/2.0)+z], type=typeS1))
                tape_parts.append(system.part.add(pos=[40.0,y,(system.box_l[2]/2.0)+z], type=typeS1))
            else:
                tape_parts.append(system.part.add(pos=[-1.0,y,(system.box_l[2]/2.0)+z], type=typeS2))
                tape_parts.append(system.part.add(pos=[40.0,y,(system.box_l[2]/2.0)+z], type=typeS2))
    
    # make the non-sticky beads
    for y in ygrid:
        for z in zgrid:
            if z > 0:
                tape_parts.append(system.part.add(pos=[-2.0,y,(system.box_l[2]/2.0)+z], type=typeSN))
                tape_parts.append(system.part.add(pos=[41.0,y,(system.box_l[2]/2.0)+z], type=typeSN))
            else:
                tape_parts.append(system.part.add(pos=[-2.0,y,(system.box_l[2]/2.0)+z], type=typeSN))
                tape_parts.append(system.part.add(pos=[41.0,y,(system.box_l[2]/2.0)+z], type=typeSN))
    
    # bond the tape beads together
    springBondStructure(system,tape_parts,50.0,2.0)
    
    # non-bonded interactions between tape beads and lipid beads
    
    # treat head beads separately, always purely repulsive
    b0 = 0.5+r[0]
    b4 = 0.5+r[4]
    system.non_bonded_inter[0,typeS1].lennard_jones.set_params(
                                epsilon=lj_eps, sigma=b0,
                                cutoff=np.power(2.0, 1.0/6.0)*b0, shift="auto")
    system.non_bonded_inter[0,typeS2].lennard_jones.set_params(
                                epsilon=lj_eps, sigma=b0,
                                cutoff=np.power(2.0, 1.0/6.0)*b0, shift="auto")
    system.non_bonded_inter[0,typeSN].lennard_jones.set_params(
                                epsilon=lj_eps, sigma=b0,
                                cutoff=np.power(2.0, 1.0/6.0)*b0, shift="auto")
    
    system.non_bonded_inter[4,typeS1].lennard_jones.set_params(
                                epsilon=lj_eps, sigma=b4,
                                cutoff=np.power(2.0, 1.0/6.0)*b4, shift="auto")
    system.non_bonded_inter[4,typeS2].lennard_jones.set_params(
                                epsilon=lj_eps, sigma=b4,
                                cutoff=np.power(2.0, 1.0/6.0)*b4, shift="auto")
    system.non_bonded_inter[4,typeSN].lennard_jones.set_params(
                                epsilon=lj_eps, sigma=b4,
                                cutoff=np.power(2.0, 1.0/6.0)*b4, shift="auto")
    
    # tail beads
    for i in range(1,4):
        bi = 0.5+r[i]
        system.non_bonded_inter[i,typeS1].lennard_jones_cos2.set_params(
                                    epsilon=lj_eps, sigma=bi,
                                    width=wc, offset=0.)
        system.non_bonded_inter[i,typeS2].lennard_jones.set_params(
                                    epsilon=lj_eps, sigma=bi,
                                    cutoff=np.power(2.0, 1.0/6.0)*bi, shift="auto")
        system.non_bonded_inter[i,typeSN].lennard_jones.set_params(
                                    epsilon=lj_eps, sigma=bi,
                                    cutoff=np.power(2.0, 1.0/6.0)*bi, shift="auto")
    for i in range(5,8):
        bi = 0.5+r[i]
        system.non_bonded_inter[i,typeS1].lennard_jones.set_params(
                                    epsilon=lj_eps, sigma=bi,
                                    cutoff=np.power(2.0, 1.0/6.0)*bi, shift="auto")
        system.non_bonded_inter[i,typeS2].lennard_jones_cos2.set_params(
                                    epsilon=lj_eps, sigma=bi,
                                    width=wc, offset=0.)
        system.non_bonded_inter[i,typeSN].lennard_jones.set_params(
                                    epsilon=lj_eps, sigma=bi,
                                    cutoff=np.power(2.0, 1.0/6.0)*bi, shift="auto")
    
    # return particle handle list
    return tape_parts

if __name__ == "__main__":
    main()
