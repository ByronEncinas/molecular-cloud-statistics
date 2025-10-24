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

cycle = 0 

reduction_factor = []
prev_time = 0.0

file_list = [sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5'))[::spacing], sorted(glob.glob('arepo_data/ambipolar_diffusion/*.hdf5'))[::spacing]]

if len(file_list) == 0:
    print("No files to process.")
    exit()

for list in file_list:

    for fileno, filename in enumerate(list[::-1]):

        
        data = h5py.File(filename, 'r')
        header_group = data['Header']
        time_value = header_group.attrs['Time']
        snap = filename.split('/')[-1].split('.')[0][-3:]
        if 'amb' in filename:
            typpe = 'amb'
        else:
            typpe = 'ideal'

        Boxsize = data['Header'].attrs['BoxSize']

        VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
        Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
        Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
        Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
        Velocities = np.asarray(data['PartType0']['Velocities'], dtype=FloatType)
        Momentums = Mass[:, np.newaxis] * Velocities

        Volume   = Mass/Density
        time_code_units = time_value*myrs_to_code_units
        delta_time_seconds = (time_value-prev_time)*seconds_in_myr
        
        xc = Pos[:, 0]
        yc = Pos[:, 1]
        zc = Pos[:, 2]

        region_radius = 0.5
            
        if fileno == 0:
            CloudCord = Pos[np.argmax(Density), :]
            PeakDensity = Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3
            print(PeakDensity)
            with open(f"./{typpe}_cloud_tracjetory.txt", "w") as file:
                file.write("snap,time_value,CloudCord_X,CloudCord_Y,CloudCord_Z,Peak_Density\n")
                file.write(f"{snap},{time_value},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")
        else:
            cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)
            Radius = np.sqrt((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2)

            if np.any(cloud_sphere):
                CloudCord = Pos[cloud_sphere][np.argmax(Density[cloud_sphere]), :]
                PeakDensity = Density[cloud_sphere][np.argmax(Density[cloud_sphere])]*gr_cm3_to_nuclei_cm3            
                print(PeakDensity)

            with open(f"./{typpe}_cloud_tracjetory.txt", "a") as file:
                file.write(f"{snap},{time_value},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")

        prev_time = time_value
        
        continue