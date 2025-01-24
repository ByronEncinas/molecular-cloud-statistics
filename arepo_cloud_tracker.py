"""
Cloud Tracker evaluates R statistics in time evolution while following the relative position with the Box center 
of the molecular cloud core.

python3 arepo_cloud_tracker.py [N] [r_upperbound] [# lines] []
"""
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import healpy as hp
import numpy as np
import random
import glob
import time
import h5py
import json
import sys
import os
import yt
from library import *
import logging

# Set yt logging to show only warnings and errors
yt.funcs.mylog.setLevel(logging.WARNING)

start_time = time.time()
FloatType = np.float64
IntType = np.int32

# python3 arepo_cloud_tracker.py 5_000 0.3 500 2

if len(sys.argv)>5:
	N=int(sys.argv[1])
	rloc_boundary=float(sys.argv[2])
	max_cycles   =int(sys.argv[3])
	spacing = int(sys.argv[4])
	how_many = int(sys.argv[5])
else:
    N             =4_000
    rloc_boundary =0.5
    max_cycles    =20
    spacing      = 1
    how_many     = 50

print(sys.argv[1:])
cycle = 0 

reduction_factor = []
prev_time = 0.0

"""  B. Jesus Velazquez """

def get_along_lines(x_init, dir = ''):

    dx = 0.5

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
        print(np.log10(dens[:3]))
        
        #print(threshold)
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
                with open(f'{dir}iso_radius_vectors{snap}.dat', 'a') as file: 
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
        print(np.log10(dens[:3]))
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
                with open(f'{dir}iso_radius_vectors{snap}.dat', 'a') as file: 
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
	
    for _n in range(m): # Iterate over the first dimension
        prev = radius_vector[0, _n, :]
        for k in range(magnetic_fields.shape[0]):  # Iterate over the first dimension
            radius_to_origin[k, _n] = magnitude(radius_vector[k, _n, :])
            cur = radius_vector[k, _n, :]
            diff_rj_ri = magnitude(cur, prev)
            trajectory[k,_n] = trajectory[k-1,_n] + diff_rj_ri            
            prev = radius_vector[k, _n, :]
    
    trajectory[0,:]  = 0.0

    radius_vector   *= 1.0* 3.086e+18                                # from Parsec to cm
    trajectory      *= 1.0* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)
    volumes_all     *= 1.0#/(3.086e+18**3) 

    return bfields[0,:], radius_vector, trajectory, magnetic_fields, numb_densities, volumes_all, radius_to_origin, [threshold, threshold_rev]

#/ideal_mhd
#file_list = sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5'))[::spacing]
file_list = sorted(glob.glob('arepo_data/ambipolar_diffusion/*.hdf5'))[::spacing]
#file_list = sorted(glob.glob('arepo_data/*.hdf5'))[::spacing]

if len(file_list) == 0:
    print("No files to process.")
    exit()

for fileno, filename in enumerate(file_list[::-1][0:how_many]):
    
    data = h5py.File(filename, 'r')
    header_group = data['Header']
    time_value = header_group.attrs['Time']
    snap = filename.split('/')[-1].split('.')[0][-3:]
    typpe = filename.split('/')[-1].split('_')[0]
    parent_folder = "cloud_tracker_slices/"+typpe 
    children_folder = os.path.join(parent_folder, 'ct_'+snap)
    print(children_folder)
    os.makedirs(children_folder, exist_ok=True)
    Boxsize = data['Header'].attrs['BoxSize']
    VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
    Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
    Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
    Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
    Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
    Velocities = np.asarray(data['PartType0']['Velocities'], dtype=FloatType)
    Momentums = Mass[:, np.newaxis] * Velocities
    InternalEnergy = np.asarray(data['PartType0']['InternalEnergy'], dtype=FloatType)
    Pressure = np.asarray(data['PartType0']['Pressure'], dtype=FloatType)
    Bfield_grad = np.zeros((len(Pos), 9))
    Density_grad = np.zeros((len(Density), 3))
    Volume   = Mass/Density
    time_code_units = time_value*myrs_to_code_units
    delta_time_seconds = (time_value-prev_time)*seconds_in_myr
    xc = Pos[:, 0]
    yc = Pos[:, 1]
    zc = Pos[:, 2]
    region_radius = 8

    if fileno == 0:
        # Open the file for the first time (when fileno = 0)
        # Initialize CloudCord based on the max density position
        CloudCord = Pos[np.argmax(Density), :]
        #with open(f"cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt", "a") as file:
        with open(f"cloud_tracker_slices/_cloud_trajectory.txt", "w") as file:
            file.write("snap,time_value,CloudCord_X,CloudCord_Y,CloudCord_Z,CloudVel_X,CloudVel_Y,CloudVel_Z\n")
            file.write(f"{snap},{np.round(time_value,5)},{np.round(CloudCord[0],8)},{np.round(CloudCord[1],8)},{np.round(CloudCord[2],8)},{np.round(0.0,8)}, {np.round(0.0,8)}, {np.round(0.0,8)}\n")
    else:
        # isolate values surrounding cloud
        cloud_sphere = ((xc-CloudCord[0])**2 + (yc-CloudCord[1])**2 + (zc-CloudCord[2])**2 < region_radius)
        
        CloudVelocity = np.sum(Momentums[cloud_sphere, :], axis=0)/np.sum(Mass[cloud_sphere])

        delta_time_seconds = abs(time_value-prev_time) * seconds_in_myr

        UpdatedCord = CloudCord - CloudVelocity * km_to_parsec * delta_time_seconds

        #cloud_sphere = (xc-UpdatedCord[0])**2 + (yc-UpdatedCord[1])**2 + (zc-UpdatedCord[2])**2 < region_radius

        CloudCord = Pos[np.argmax(Density[cloud_sphere]), :] #UpdatedCord.copy() 

        #with open(f"cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt", "a") as file:
        with open(f"cloud_tracker_slices/_cloud_trajectory.txt", "a") as file:
            file.write(f"{snap},{np.round(time_value,5)},{np.round(CloudCord[0],8)},{np.round(CloudCord[1],8)},{np.round(CloudCord[2],8)},{np.round(CloudVelocity[0],8)}, {np.round(CloudVelocity[1],8)}, {np.round(CloudVelocity[2],8)}\n")
    
    print(CloudCord, delta_time_seconds, filename)
    
    prev_time = time_value

    ds = yt.load(filename)

    # Access the all_data object
    ad = ds.all_data()

    # Create the slice plot at z = CloudCord[2]
    sp = yt.SlicePlot(
        ds, 
        'z', 
        ('gas', 'density'), 
        center=[CloudCord[0], CloudCord[1], CloudCord[2]],
        width = 25
    )

    sp.annotate_marker(
        [CloudCord[0], CloudCord[1], CloudCord[2]],
        marker='x',
        color='red',
        s=100
    )

    # Annotate the plot with timestamp and scale
    sp.annotate_timestamp(redshift=False)
    sp.annotate_scale()

    # Save the plot as a PNG file {fileno}-{filename.split('/')[-1]}
    sp.save(os.path.join(parent_folder, f"{typpe}_{snap}_slice_z.png"))

    continue