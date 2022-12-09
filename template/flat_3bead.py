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

jobname = "flat_3bead"

# System parameters
#############################################################
# Main simulation parameters:
skin          = 0.4
time_step     = 0.005
temperature       = 1.4
integrator_steps  = 200
sampling_interval = 10
iterations        = 5000

# Warmup parameters:
warm_tstep  = 0.0005
warm_steps  = 10
min_dist    = 0.90

# Cooke lipid model parameters
bhh = 0.95
bht = 0.95
btt = 1.
hh_cut = pow(2., 1./6) * bhh
ht_cut = pow(2., 1./6) * bht
tt_cut = pow(2., 1./6) * btt
cshift = 1./4.
lj_eps = 1.0
cshift = 1./4.
wc = 1.6

def main():
    #############################################################
    # Set Up                                                    #
    #############################################################
    system = espressomd.System(box_l = [12.4,12.4,20.0]) #periodic BC by default
    system.set_random_state_PRNG()
    system.time_step = warm_tstep
    system.cell_system.skin = skin
    
    #############################################################
    # Membrane                                                  #
    #############################################################
    typeA = {"Head":0,"Mid":1,"Tail":1}
    typeB = {"Head":0,"Mid":1,"Tail":1}
    numA = 128
    numB = 128
    
    uPos, dPos, uAngle, dAngle = flatBilayer(system, nTop=numA, nBot=numB, z0=2.0, verbose=verb)
    # to set up other membrane geometries, replace the above with other setup functions:
    # = cylindricalBilayer(system, r, numOut, numIn, z0o=2.0, z0i=2.0, verbose=False)
    # = buckledBilayer(system, L, nTop, nBot=None, z0=2.0, verbose=False)
    
    # Create lists of lipid handles of desired types
    ulipids = [Lipid(system,lipidType=typeA) for i in range(numA)]
    dlipids = [Lipid(system,lipidType=typeB) for i in range(numB)]
    
    placeLipids(ulipids, dlipids, uPos, dPos, uAngle, dAngle)
    
    # combine lipid handles into single array
    lipids = ulipids + dlipids
    
    if verb:
        print("Total number of lipids: " + str(len(lipids)) )
        print("kT={}".format(temperature))
    
    #############################################################
    # Non bonded Interactions between the lipid beads           #
    #############################################################

    system.non_bonded_inter[0, 0].lennard_jones.set_params(
            epsilon=lj_eps, sigma=bhh,
            cutoff=hh_cut, shift=cshift)

    system.non_bonded_inter[0, 1].lennard_jones.set_params(
            epsilon=lj_eps, sigma=bht,
            cutoff=ht_cut, shift=cshift)

    # Attractive Tail-Tail
    system.non_bonded_inter[1, 1].lennard_jones_cos2.set_params(
            epsilon=lj_eps, sigma=btt,
            width=wc, offset=0.)
    
    #############################################################
    #  Output Files                                             #
    #############################################################
    try:
        os.mkdir(jobname)
    except OSError:
        print("Directory {} already exists or could not be created.".format(jobname))
    
    with open(jobname+"/trajectory.vtf", "w") as vtf_fp, open(jobname+"/energy.txt","w") as en_fp, \
        open(jobname+"/box.txt","w") as box_fp, open(jobname+"/flipflop.txt","w") as ff_fp:
        
        # write structure block header
        vtf.writevsf(system, vtf_fp)
        # write initial position coordinate block
        vtf.writevcf(system, vtf_fp)
        #NOTE: WRITING INITIAL CONFIG FIRST
        
        #############################################################
        #  Warmup Integration                                       #
        #############################################################
        
        if verb:
            print("Starting warmup integration:")
            print("Proceeding to main integration if minimum distance > {} sigma".format(min_dist) )
        
        # set the integrator to capped gradient descent
        system.integrator.set_steepest_descent(f_max=0, gamma=10, max_displacement=min_dist*0.01)
        
        # gradient descent until particles are separated by at least min_dist
        act_min_dist = system.analysis.min_dist()
        while act_min_dist < min_dist:
            system.integrator.run(warm_steps)
            act_min_dist = system.analysis.min_dist()
            if verb: print("Running steepest descent warm-up, current min_dist = {}".format(act_min_dist))
        
        system.integrator.set_vv()   # Switch to velocity-verlet integrator
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
                # Write coordinates out to trajectory file
                vtf.writevcf(system, vtf_fp)
                # Write box into trajectory timestep
                vtf_fp.write("unitcell {} {} {}\n".format(*system.box_l) )
                
                e = system.analysis.energy()
                en_fp.write("{},{},{},{}\n".format(e["total"],e["kinetic"],e["bonded"],e["non_bonded"]))
                
                box_fp.write("{},{},{}\n".format(*system.box_l) )
                
                f = leafletContent(system, lipids, typeA, typeB)
                ff_fp.write("{},{},{},{},{}\n".format(*f))
                
                en_fp.flush()
                box_fp.flush()
                vtf_fp.flush()
                ff_fp.flush()
                print("Completed {} iterations (tau={})".format(i,i*integrator_steps*time_step))
    
    elapsed = time.time() - start_time
    print(jobname + " Done. Elapsed time {} seconds".format(elapsed))
    

if __name__ == "__main__":
    main()

