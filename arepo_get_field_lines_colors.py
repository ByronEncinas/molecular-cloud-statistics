from collections import defaultdict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy import spatial
import healpy as hp
import numpy as np
import random
import shutil
import h5py
import json
import sys
import os

from library import *

import time

start_time = time.time()

"""  
Using Margo Data

Analysis of reduction factor

$$N(s) 1 - \sqrt{1-B(s)/B_l}$$

Where $B_l$ corresponds with (in region of randomly choosen point) the lowest between the highest peak at both left and right.
where $s$ is a random chosen point at original 128x128x128 grid.

1.- Randomly select a point in the 3D Grid. 
2.- Follow field lines until finding B_l, if non-existent then change point.
3.- Repeat 10k times
4.- Plot into a histogram.

contain results using at least 20 boxes that contain equally spaced intervals for the reduction factor.

# Calculating Histogram for Reduction Factor in Randomized Positions in the 128**3 Cube 

"""

"""
Parameters

- [N] default total number of steps in the simulation
- [dx] default 4/N of the rloc_boundary (radius of spherical region of interest) variable

"""
FloatType = np.float64
IntType = np.int32

if len(sys.argv)>2:
	# first argument is a number related to rloc_boundary
	N=int(sys.argv[1])
	rloc_boundary=float(sys.argv[2])
	max_cycles   =int(sys.argv[3])
else:
    N            =100
    rloc_boundary=256   # rloc_boundary for boundary region of the cloud
    max_cycles   =1

"""  B. Jesus Velazquez """

snap = '3000'
filename = 'arepo_data/snap_ambipolar' + snap + '.hdf5'

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)

# Initialize gradients
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

Volume   = Mass/Density

"""  
Attribute: UnitLength_in_cm = 3.086e+18
Attribute: UnitMass_in_g = 1.99e+33
Attribute: UnitVelocity_in_cm_per_s = 100000.0

Name: PartType0/Coordinates
    Attribute: to_cgs = 3.086e+18
Name: PartType0/Density
    Attribute: to_cgs = 6.771194847794873e-23
Name: PartType0/Masses
    Attribute: to_cgs = 1.99e+33
Name: PartType0/Velocities
    Attribute: to_cgs = 100_000.0
Name: PartType0/MagneticField

"""

#Center= 0.5 * Boxsize * np.ones(3) # Center
#Center = np.array( [91,       -110,          -64.5]) #117
#Center = np.array( [96.6062303,140.98704002, 195.78020632]) #117
Center = Pos[np.argmax(Density),:] #430
print("Center before Centering", Center)

VoronoiPos-=Center
Pos-=Center

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

def get_along_lines(x_init):

    dx = 1.0

    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))
    volumes   = np.zeros((N+1,m))
    threshold = np.zeros((m,)).astype(int) # one value for each

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
    volumes_rev   = np.zeros((N+1,m))
    threshold_rev = np.zeros((m,)).astype(int) # one value for each

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init

    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3

    k=0

    mask = dens > 100 # True if not finished
    un_masked = np.logical_not(mask)

    while np.any(mask):

        # Create a mask for values that are 10^2 N/cm^3 above the threshold
        mask = dens > 100 # 1 if not finished
        un_masked = np.logical_not(mask) # 1 if finished

        aux = x[un_masked]

        x, bfield, dens, vol = Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        if len(threshold[un_masked]) != 0:
            unique_unmasked_max_threshold = np.max(np.unique(threshold[un_masked]))
            max_threshold = np.max(threshold)
        else:
            unique_unmasked_max_threshold = np.max(threshold)
            max_threshold = np.max(threshold)
        
        x[un_masked] = aux
        
        #print(threshold)
        #
        # print(max_threshold, unique_unmasked_max_threshold)

        line[k+1,:,:]    = x
        volumes[k+1,:]   = vol
        bfields[k+1,:]   = bfield
        densities[k+1,:] = dens

        step_diff = max_threshold-unique_unmasked_max_threshold
        
        order_clause = step_diff >= 1_000
        percentage_clause = np.sum(un_masked)/len(mask) > 0.8

        if np.all(un_masked) or (order_clause and percentage_clause): 
            if (order_clause and percentage_clause):
                with open(f'lone_run_radius_vectors{snap}.dat', 'a') as file: 
                    file.write(f"{order_clause} and {percentage_clause} of file {filename}\n")
                    file.write(f"{x_init[mask]}\n")
                print("80% of lines have concluded ")
            else:
                print("All values are False: means all crossed the threshold")
            break    

        k += 1
    
    threshold = threshold.astype(int)
    
    x = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]

    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    print(line_rev.shape)

    k=0

    mask_rev = dens > 100
    un_masked_rev = np.logical_not(mask_rev)
    
    while np.any((mask_rev)):

        mask_rev = dens > 100 
        un_masked_rev = np.logical_not(mask_rev)

        aux = x[un_masked_rev]

        x, bfield, dens, vol = Heun_step(x, -dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold_rev += mask_rev.astype(int)

        if len(threshold_rev[un_masked_rev]) != 0:
            unique_unmasked_max_threshold = np.max(np.unique(threshold_rev[un_masked_rev]))
            max_threshold = np.max(threshold_rev)
        else:
            unique_unmasked_max_threshold = np.max(threshold_rev)
            max_threshold = np.max(threshold_rev)

        #print(max_threshold, unique_unmasked_max_threshold)
        x[un_masked_rev] = aux

        line_rev[k+1,:,:] = x
        volumes_rev[k+1,:] = vol
        bfields_rev[k+1,:] = bfield
        densities_rev[k+1,:] = dens 
                    
        step_diff = max_threshold-unique_unmasked_max_threshold
        
        order_clause = step_diff >= 1_000
        percentage_clause = np.sum(un_masked_rev)/len(mask_rev) > 0.8

        if np.all(un_masked_rev) or (order_clause and percentage_clause):
            if (order_clause and percentage_clause):
                with open(f'lone_run_radius_vectors{snap}.dat', 'a') as file: 
                    file.write(f"{order_clause} and {percentage_clause} of file {filename}\n")
                    file.write(f"{x_init[mask_rev]}\n")
                print("80% of lines have concluded ")
            else:
                print("All values are False: means all crossed the threshold")
            break

        k += 1

    updated_mask = np.logical_not(np.logical_and(mask, mask_rev))
    
    threshold = threshold[updated_mask].astype(int)
    threshold_rev = threshold_rev[updated_mask].astype(int)

    # Apply updated_mask to the second axis of (N+1, m, 3) or (N+1, m) arrays
    line = line[:, updated_mask, :]  # Mask applied to the second dimension (m)
    volumes = volumes[:, updated_mask]  # Assuming volumes has shape (m,)
    bfields = bfields[:, updated_mask]  # Mask applied to second dimension (m)
    densities = densities[:, updated_mask]  # Mask applied to second dimension (m)

    # Apply to the reverse arrays in the same way
    line_rev = line_rev[:, updated_mask, :]
    volumes_rev = volumes_rev[:, updated_mask]
    bfields_rev = bfields_rev[:, updated_mask]
    densities_rev = densities_rev[:, updated_mask]
    
    radius_vector = np.append(line_rev[::-1, :, :], line, axis=0)
    magnetic_fields = np.append(bfields_rev[::-1, :], bfields, axis=0)
    numb_densities = np.append(densities_rev[::-1, :], densities, axis=0)
    volumes_all = np.append(volumes_rev[::-1, :], volumes, axis=0)

    #gas_densities   *= 1.0* 6.771194847794873e-23                      # M_sol/pc^3 to gram/cm^3
    #numb_densities   = gas_densities.copy() * 6.02214076e+23 / 1.00794 # from gram/cm^3 to Nucleus/cm^3

    # Initialize trajectory and radius_to_origin with the same shape
    trajectory = np.zeros_like(magnetic_fields)
    radius_to_origin = np.zeros_like(magnetic_fields)

    print("Magnetic fields shape:", magnetic_fields.shape)
    print("Radius vector shape:", radius_vector.shape)
    print("Numb densities shape:", numb_densities.shape)

    m = magnetic_fields.shape[1]
    print("Surviving lines: ", m, "out of: ", max_cycles)

    for _n in range(m):  # Iterate over the second dimension
        start_idx = (magnetic_fields.shape[0] - threshold_rev[_n]).astype(int) 
        prev = radius_vector[0, _n, :]
        for k in range(magnetic_fields.shape[0]):  # Iterate over the first dimension
            radius_to_origin[k, _n] = np.linalg.norm(radius_vector[k, _n, :])
            cur = radius_vector[k, _n, :]
            diff_rj_ri = np.linalg.norm(cur - prev)*3.086e+18  # Calculate the difference between consecutive points
            if k == 0:
                trajectory[k, _n] = 0.0  # Ensure the starting point of trajectory is zero
            else:
                trajectory[k, _n] = trajectory[k-1, _n] + diff_rj_ri
            prev = cur  # Update `prev` to the current point
            print("Trajectory at step", k, ":", trajectory[k, _n])

        #trajectory[:,_n] = trajectory[:,_n] - trajectory[start_idx,_n] #np.zeros((m,))
    
    radius_vector   *= 1.0* 3.086e+18                                # from Parsec to cm
    #trajectory      *= 1.0* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)
    volumes_all     *= 1.0#/(3.086e+18**3) 

    return bfields[0,:], radius_vector, trajectory, magnetic_fields, numb_densities, volumes_all, radius_to_origin, [threshold, threshold_rev]

if sys.argv[-1] == "input":

    unique_vectors = set()  # Use a set to store unique vectors

    with open('lone_run_radius_vectors.dat', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('['):  # Assuming x_init vectors are represented as arrays in brackets
                # Convert the string representation of the array to a numpy array
                vector = np.fromstring(line.strip().strip('[]'), sep=',')
                # Convert to tuple (so it can be added to a set) and add to unique_vectors
                unique_vectors.add(tuple(vector))

    # Step 2: Convert set back to a list of unique numpy arrays
    x_init = np.array([np.array(vec) for vec in unique_vectors])

    with open('output', 'w') as file:
        file.write(f"Make sure the initial radius vectors correspond with the file that will provide the\n")
        file.write(f"Magnetic field that will be use to trace field lines\n")
        file.write(f"{filename}\n")
        file.write(f"Cores Used: {os.cpu_count()}\n")
        file.write(f"Steps in Simulation: {2 * N}\n")
        file.write(f"max_cycles         : {len(x_init[0,:,0])}\n")
        file.write(f"Boxsize (Pc)       : {Boxsize} Pc\n")
        file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")
else:
    rloc_center      = np.array([float(random.uniform(0,rloc_boundary)) for l in range(max_cycles)])
    nside = 1_000     # sets number of cells sampling the spherical boundary layers = 12*nside**2
    npix  = 12 * nside ** 2
    ipix_center       = np.arange(npix)
    xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix_center)

    xx = np.array(random.sample(list(xx), max_cycles))
    yy = np.array(random.sample(list(yy), max_cycles))
    zz = np.array(random.sample(list(zz), max_cycles))

    m = len(zz) # amount of values that hold which_up_down

    x_init = np.zeros((m,3))

    x_init[:,0]      = rloc_center * xx[:]
    x_init[:,1]      = rloc_center * yy[:]
    x_init[:,2]      = rloc_center * zz[:]

    lmn = N

    with open('output', 'w') as file:
        file.write(f"{filename}\n")
        file.write(f"Cores Used: {os.cpu_count()}\n")
        file.write(f"Steps in Simulation: {2 * N}\n")
        file.write(f"rloc_boundary (Pc) : {rloc_boundary}\n")
        file.write(f"rloc_center (Pc)   :\n {rloc_center}\n")
        file.write(f"x_init (Pc)        :\n {x_init}\n")
        file.write(f"max_cycles         : {max_cycles}\n")
        file.write(f"Boxsize (Pc)       : {Boxsize} Pc\n")
        file.write(f"Center (Pc, Pc, Pc): {Center[0]}, {Center[1]}, {Center[2]} \n")
        file.write(f"Posit Max Density (Pc, Pc, Pc): {Pos[np.argmax(Density), :]}\n")
        file.write(f"Smallest Volume (Pc^3)   : {Volume[np.argmin(Volume)]} \n")
        file.write(f"Biggest  Volume (Pc^3)   : {Volume[np.argmax(Volume)]}\n")
        file.write(f"Smallest Density (M☉/Pc^3)  : {Density[np.argmin(Volume)]} \n")
        file.write(f"Biggest  Density (M☉/Pc^3) : {Density[np.argmax(Volume)]}\n")
        file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")

print("Cores Used         : ", os.cpu_count())
print("Steps in Simulation: ", 2*N)
print("rloc_boundary      : ", rloc_boundary)
print("rloc_center        :\n ", rloc_center)
print("max_cycles         : ", max_cycles)
print("Boxsize            : ", Boxsize) # 256
print("Center             : ", Center) # 256
print("Posit Max Density  : ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume    : ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume    : ", Volume[np.argmax(Volume)]) # 256
print(f"Smallest Density  : {Density[np.argmin(Density)]}")
print(f"Biggest  Density  : {Density[np.argmax(Density)]}")
"""
x_init = np.array([[ 0.03518049, -0.06058562,  0.09508827],
                   [-0.08827144,  0.07445224, -0.01678605],
                   [ 0.11605630,  0.24466445, -0.32513439],
                   [-0.01082023, -0.02556539, -0.00694829],
                   [-0.19161783,  0.10030747,  0.14942809]])
"""
__, radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th = get_along_lines(x_init)

m = magnetic_fields.shape[1]

threshold, threshold_rev = th

print("Elapsed Time: ", (time.time() - start_time)/60.)

for i in range(m):

    print(f"{threshold[i]} - {threshold_rev[i]}")
    _from = N+1 - threshold_rev[i]
    _to   = N+1 + threshold[i]
    print(f"{_from} - {_to}")

    mag_field    = magnetic_fields[_from:_to,i]
    pos_vector   = radius_vector[_from:_to,i,:]
    s_coordinate = trajectory[_from:_to,i] - trajectory[_from,i]
    numb_density = numb_densities[_from:_to,i]
    volume       = volumes[_from:_to,i]
    """ 
    where = (magnetic_fields[:,i] != 0.0)

    mag_field    = magnetic_fields[where,i]
    pos_vector   = radius_vector[where,i,:]
    s_coordinate = trajectory[where,i]
    numb_density = numb_densities[where,i]
    volume       = volumes[where,i]
    """    
    pocket, global_info = pocket_finder(mag_field, i, plot=True) # this plots

    np.save(f"arepo_npys/ArePositions{i}.npy", pos_vector)
    np.save(f"arepo_npys/ArepoTrajectory{i}.npy", s_coordinate)
    np.save(f"arepo_npys/ArepoNumberDensities{i}.npy", numb_density)
    np.save(f"arepo_npys/ArepoMagneticFields{i}.npy", mag_field)
	
    
    print(f"finished line {i+1}/{max_cycles}",(time.time()-start_time)/60)

    if True:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        #plt.ticklabel_format(axis='both', style='sci')
        
        axs[0,0].plot(mag_field, linestyle="--", color="m")
        #axs[0,0].scatter(trajectory[:,i], magnetic_fields[:,i], marker="+", color="m")
        #axs[0,0].scatter(trajectory[lmn,i], magnetic_fields[lmn,i], marker="x", color="black")
        axs[0,0].set_xlabel("s (cm)")
        axs[0,0].set_ylabel("$B(s)$ $\mu$G (cgs)")
        axs[0,0].set_title("Magnetic FIeld")
        axs[0,0].grid(True)
		
        axs[0,1].plot(s_coordinate, linestyle="--", color="m")
        #axs[0,1].scatter(trajectory[:,i], marker="+", color="m")
        #axs[0,1].scatter(trajectory[lmn,i], radius_to_origin[lmn,i], marker="x", color="black")
        axs[0,1].set_xlabel("s (cm)")
        axs[0,1].set_ylabel("$r$ cm (cgs)")
        axs[0,1].set_title("Distance Away of MaxDensityCoord $r$ ")
        axs[0,1].grid(True)

        axs[1,0].plot(numb_density, linestyle="--", color="m")
        #axs[1,0].scatter(trajectory[:,i], numb_densities[:,i], marker="+", color="m")
        #axs[1,0].scatter(trajectory[lmn,i], numb_densities[lmn,i], marker="x", color="black")
        axs[1,0].set_yscale('log')
        axs[1,0].set_xlabel("s (cm)")
        axs[1,0].set_ylabel("$N_g(s)$ cm^-3 (cgs)")
        axs[1,0].set_title("Number Density (Nucleons/cm^3) ")
        axs[1,0].grid(True)
		
        axs[1,1].plot(volume, linestyle="-", color="black")
        axs[1,1].set_yscale('log')
        axs[1,1].set_xlabel("steps")
        axs[1,1].set_ylabel("$V(s) cm^3 $ (cgs)")
        axs[1,1].set_title("Cells Volume along Path")
        axs[1,1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"field_shapes/shapes_{i}.png")

        # Close the plot
        plt.close(fig)
		
if True:
    ax = plt.figure().add_subplot(projection='3d')

    for k in range(m):
        x=radius_vector[:,k,0]/ 3.086e+18                           
        y=radius_vector[:,k,1]/ 3.086e+18
        z=radius_vector[:,k,2]/ 3.086e+18
        
        for l in range(len(x)):
            ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color="m",linewidth=0.3)

        #ax.scatter(x_init[0], x_init[1], x_init[2], marker="v",color="m",s=10)
        ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
        ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
            
    zoom = np.max(radius_to_origin)
    ax.set_xlim(-zoom,zoom)
    ax.set_ylim(-zoom,zoom)
    ax.set_zlim(-zoom,zoom)
    ax.set_xlabel('x [Pc]')
    ax.set_ylabel('y [Pc]')
    ax.set_zlabel('z [Pc]')
    ax.set_title('Magnetic field morphology')

    plt.savefig(f'field_shapes/MagneticFieldTopology.png', bbox_inches='tight')