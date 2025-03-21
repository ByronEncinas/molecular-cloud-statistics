"""
Cloud Tracker evaluates R statistics in time evolution while following the relative position with the Box center 
of the molecular cloud core.

python3 arepo_cloud_tracker.py [N] [r_upperbound] [# lines] []
"""
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats
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

if len(sys.argv)>2:
	spacing = int(sys.argv[1])
	how_many = int(sys.argv[2])
else:

    spacing      = 1
    how_many     = 500

print(sys.argv[1:])
cycle = 0 

reduction_factor = []
prev_time = 0.0

#file_list = sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5'))[::spacing]
#file_list = [sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5'))[::spacing], sorted(glob.glob('arepo_data/ambipolar_diffusion/*.hdf5'))[::spacing]]
file_list = [sorted(glob.glob('arepo_data/*.hdf5'))[::spacing]]

print(file_list)

if len(file_list) == 0:
    print("No files to process.")
    exit()

for list in file_list:
    print(list)

    for fileno, filename in enumerate(list[::-1]):

        print("_>", fileno, filename)
        
        data = h5py.File(filename, 'r')
        header_group = data['Header']
        time_value = header_group.attrs['Time']
        snap = filename.split('/')[-1].split('.')[0][-3:]
        if 'amb' in filename:
            typpe = 'amb'
        else:
            typpe = 'ideal'
        print(typpe, snap)
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
        region_radius = 1
            
        if fileno == 0:
            # Initialize CloudCord based on the max density position
            CloudCord = Pos[np.argmax(Density), :]
            PeakDensity = Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3
            with open(f"cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt", "w") as file:
            #with open(f"cloud_tracker_slices/_cloud_trajectory.txt", "w") as file:
                file.write("snap,time_value,CloudCord_X,CloudCord_Y,CloudCord_Z,CloudVel_X,CloudVel_Y,CloudVel_Z,Peak_Density\n")
                file.write(f"{snap},{time_value},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},0.0,0.0,0.0,{PeakDensity}\n")
        else:
            # Isolate values surrounding CloudCord
            cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)
            
            # Find the density peak within the sphere
            if np.any(cloud_sphere):  # Ensure there are particles within the region
                CloudCord = Pos[cloud_sphere][np.argmax(Density[cloud_sphere]), :]
            else:
                print(f"Warning: No particles found within region_radius of {region_radius} around CloudCord.")
            
            # Compute CloudVelocity
            CloudVelocity = np.sum(Momentums[cloud_sphere], axis=0) / np.sum(Mass[cloud_sphere])

            # Update CloudCord based on velocity and elapsed time
            delta_time_seconds = abs(time_value - prev_time) * seconds_in_myr
            UpdatedCord = CloudCord - CloudVelocity * km_to_parsec * delta_time_seconds

            # Recompute cloud sphere around UpdatedCord and find the new density peak
            #cloud_sphere = ((xc - UpdatedCord[0])**2 + (yc - UpdatedCord[1])**2 + (zc - UpdatedCord[2])**2 < region_radius**2)
            if np.any(cloud_sphere):
                CloudCord = Pos[cloud_sphere][np.argmax(Density[cloud_sphere]), :]
                PeakDensity = Density[cloud_sphere][np.argmax(Density[cloud_sphere])]*gr_cm3_to_nuclei_cm3
            else:
                print(f"Warning: No particles found within updated region_radius of {region_radius} around UpdatedCord.")
            print(CloudCord)
            # Save trajectory data
            with open(f"cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt", "a") as file:
            #with open(f"cloud_tracker_slices/_cloud_trajectory.txt", "a") as file:
                file.write(f"{snap},{time_value},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{CloudVelocity[0]},{CloudVelocity[1]},{CloudVelocity[2]},{PeakDensity}\n")

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