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

if len(sys.argv)>=2:
	N             = int(sys.argv[-1])
else:
    N            = 1000

"""  B. Jesus Velazquez """

directory_path = "arepo_data/"
files = list_files(directory_path, '.hdf5')
print(files)

for filename in files:
    snap = filename.split(".")[0][-3:]
    if int(snap) != 300: # approx 2 Myrs past the Supernova blasts
        continue

    # Create the directory path
    directory_path = os.path.join("density_profiles", snap)

    # Check if the directory exists, if not create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")

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

    #Center= 0.5 * Boxsize * np.ones(3) # Center
    #Center = np.array( [91,       -110,          -64.5]) #117
    #Center = np.array( [96.6062303,140.98704002, 195.78020632]) #117
    Center = Pos[np.argmax(Density),:] #430
    print("Center before Centering", Center)

    VoronoiPos-=Center
    Pos-=Center

    xPosFromCenter = Pos[:,0]
    Pos[xPosFromCenter > Boxsize/2,0]        -= Boxsize
    VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

    # Original direction vectors
    axis = np.array([[ 1.0,  0.0,  0.0],
                    [ 0.0,  1.0,  0.0],
                    [ 0.0,  0.0,  1.0],
                    [-1.0,  0.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0, -1.0]])

    # Diagonal vectors
    diagonals = np.array([[ 1.0,  1.0,  1.0],
                        [ 1.0,  1.0, -1.0],
                        [ 1.0, -1.0,  1.0],
                        [ 1.0, -1.0, -1.0],
                        [-1.0,  1.0,  1.0],
                        [-1.0,  1.0, -1.0],
                        [-1.0, -1.0,  1.0],
                        [-1.0, -1.0, -1.0]])

    # Normalize the diagonal vectors to make them unit vectors
    unit_diagonals = diagonals / np.linalg.norm(diagonals[0])

    # Combine both arrays
    directions= np.vstack((axis, unit_diagonals))

    m = directions.shape[0]

    x_init = np.zeros((m,3))

    print(directions.shape)

    def store_in_directory():
        import os
        import shutil

        new_folder = os.join("density_profiles/" , snap)
        # Create the new arepo_npys directory
        os.makedirs(new_folder, exist_ok=True)

    def get_along_lines(x_init):
        
        line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
        bfields   = np.zeros((N+1,m))
        densities = np.zeros((N+1,m))
        volumes   = np.zeros((N+1,m))
        threshold = np.zeros((m,)).astype(int) # one value for each

        line[0,:,:]     = x_init
        x = x_init
        dummy, bfields[0,:], dens, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)
        vol = Volume[cells]
        dens = dens* gr_cm3_to_nuclei_cm3
        densities[0,:] = dens
        k=0

        while np.any((dens > 100)):

            # Create a mask for values that are still above the threshold
            mask = dens > 100

            un_masked = np.logical_not(mask)
            
            # Only update values where mask is True
            _, bfield, dens, vol = Heun_step(x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)

            dens = dens * gr_cm3_to_nuclei_cm3 
            
            vol[un_masked] = 0

            dx_vec = 0.3 * ((4 / 3) * vol / np.pi) ** (1 / 3)

            threshold += mask.astype(int)  # Increment threshold count only for values still above 100
            
            print(np.log10(dens))

            x += dx_vec[:, np.newaxis] * directions

            line[k+1,:,:]    = x
            volumes[k+1,:]   = vol
            bfields[k+1,:]   = bfield
            densities[k+1,:] = dens
            
            if np.all(un_masked):
                print("All values are False: means all crossed the threshold")
                break

            k += 1

        threshold = threshold.astype(int)
        larger_cut = np.max(threshold)
        
        # cut all of them to standard
        radius_vector   = line[:larger_cut+1,:,:]
        magnetic_fields = bfields[:larger_cut+1,:]
        numb_densities   = densities[:larger_cut+1,:]
        volumes         = volumes[:larger_cut+1,:]

        # Initialize trajectory and radius_to_origin with the same shape
        trajectory      = np.zeros_like(magnetic_fields)
        radius_to_origin= np.zeros_like(magnetic_fields)

        print("Magnetic fields shape:", magnetic_fields.shape)
        print("Radius vector shape:", radius_vector.shape)
        print("Numb densities shape:", numb_densities.shape)
    
        
        for _n in range(m):  # Iterate over the first dimension

            cut = threshold[_n] - 1
            
            # make it constant after threshold, this will make diff = 0.0 so s wont grow
            #radius_vector[cut:,:,:]  = radius_vector[cut,:,:]*0
            #magnetic_fields[cut:,:]  = magnetic_fields[cut,:]*0
            #numb_densities[cut:,:]    = numb_densities[cut,:]*0
            
            prev = radius_vector[0, _n, :]
            
            for k in range(magnetic_fields.shape[0]):  # Iterate over the first dimension

                radius_to_origin[k, _n] = magnitude(radius_vector[k, _n, :])
                cur = radius_vector[k, _n, :]
                diff_rj_ri = magnitude(cur, prev)
                trajectory[k,_n] = trajectory[k-1,_n] + diff_rj_ri            
                prev = radius_vector[k, _n, :]

        radius_vector   *= 1.0* parsec_to_cm3
        trajectory      *= 1.0* parsec_to_cm3
        magnetic_fields *= 1.0* gauss_code_to_gauss_cgs
        trajectory[0,:]  = 0.0

        return radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin

    print("Steps in Simulation: ", N)
    print("Boxsize            : ", Boxsize)
    print("Smallest Volume    : ", Volume[np.argmin(Volume)])
    print("Biggest  Volume    : ", Volume[np.argmax(Volume)])
    print(f"Smallest Density  : {Density[np.argmin(Density)]}")
    print(f"Biggest  Density  : {Density[np.argmax(Density)]}")

    radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin = get_along_lines(x_init)

    print("Elapsed Time: ", (time.time() - start_time)/60.)

    for i in range(m):

        if True:
            # Create a figure and axes for the subplot layout
            fig, axs = plt.subplots(2, 2, figsize=(8, 6))
            
            axs[0,0].plot(trajectory[:,i], magnetic_fields[:,i], linestyle="--", color="m")
            axs[0,0].scatter(trajectory[:,i], magnetic_fields[:,i], marker="+", color="m")
            axs[0,0].set_xlabel("s (cm)")
            axs[0,0].set_ylabel("$B(s)$ Gauss (cgs)")
            axs[0,0].set_title("Magnetic FIeld")
            axs[0,0].grid(True)
            
            axs[0,1].plot(trajectory[:,i], radius_to_origin[:,i], linestyle="--", color="m")
            axs[0,1].scatter(trajectory[:,i], radius_to_origin[:,i], marker="+", color="m")
            axs[0,1].set_xlabel("s (cm)")
            axs[0,1].set_ylabel("$log_{10}8n_g(s) (gr/cm^3))$  (cgs)")
            axs[0,1].set_title("Distance Away of MaxDensityCoord $r$ ")
            axs[0,1].grid(True)

            axs[1,0].plot(trajectory[:,i], numb_densities[:,i], linestyle="--", color="m")
            axs[1,0].scatter(trajectory[:,i], numb_densities[:,i], marker="+", color="m")
            axs[1,0].set_xscale('log')
            axs[1,0].set_yscale('log')
            axs[1,0].set_xlabel("s (cm)")
            axs[1,0].set_ylabel("$log_{10}8n_g(s) (gr/cm^3))$  (cgs)")
            axs[1,0].set_title("Number Density (Nucleons/cm^3) ")
            axs[1,0].grid(True)
            
            axs[1,1].plot(volumes[:,i], linestyle="--", color="black")
            axs[1,1].set_yscale('log')
            axs[1,1].set_xlabel("steps")
            axs[1,1].set_ylabel("$V(s)$ cm^3 (cgs)")
            axs[1,1].set_title("Cells Volume along Path")
            axs[1,1].grid(True)

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Save the figure
            plt.savefig(f"{directory_path}/shapes_{i}.png")

            # Close the plot
            plt.close(fig)
            
    if True:
        ax = plt.figure().add_subplot(projection='3d')
        dens_min = np.log10(np.min(numb_densities))
        dens_max = np.log10(np.max(numb_densities))

        dens_diff = dens_max - dens_min

        for k in range(m):
            x=radius_vector[:,k,0]/ 3.086e+18                                # from Parsec to cm
            y=radius_vector[:,k,1]/ 3.086e+18                                # from Parsec to cm
            z=radius_vector[:,k,2]/ 3.086e+18                                # from Parsec to cm
            
            for l in range(len(radius_vector[:,0,0])):
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color='m',linewidth=0.3)
                
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
        
        plt.savefig(f"{directory_path}/MagneticFieldTopology.png", bbox_inches='tight')
