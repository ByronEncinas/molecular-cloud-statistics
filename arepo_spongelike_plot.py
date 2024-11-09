# Import necessary libraries
from collections import defaultdict
from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
from scipy import spatial
import healpy as hp
import numpy as np
import random
import os
import h5py
import json
import sys
from library import *
import time
from mpl_toolkits.mplot3d import Axes3D

start_time = time.time()

# Specify the file path
file_path = f'cluster_outputs/redfactJAKAR/position_vector_reduction15237_300'

# Write the list data to a JSON file
with open(file_path, 'r') as json_file:
    pos_red = dict(json.load(json_file))  # [x, y, z] = R basically a 3D stochastic function

radius_vector = []
red_factor = []

for k, v in pos_red.items():
    if v == 1:
        # we only want to plot where R < 1
        continue
    
    # Convert `k` from a string representation of a list to an actual list of floats
    vals = k.split(',')
    vector = [float(vals[i]) for i in range(len(vals))]
    coords = np.array(vector)  # Parse string to list of floats
    radius_vector.append(coords)
    red_factor.append(v)

sample_size = len(pos_red.items())

# Convert to numpy arrays
radius_vector = np.array(radius_vector)
red_factor = 1.0 - np.array(red_factor)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define x, y, z for plotting
x = radius_vector[:, 0]
y = radius_vector[:, 1]
z = radius_vector[:, 2]

reduced_sample = len(red_factor)

# Scatter plot with colormap based on the distance from the origin
sc = ax.scatter(x, y, z, c=red_factor, cmap='plasma', s=6)

# Add colorbar to show the scale
plt.colorbar(sc, label=f'Reduction Factor')            
ax.set_xlabel('x [Pc]')
ax.set_ylabel('y [Pc]')
ax.set_zlabel('z [Pc]')
ax.set_title(f'R < 1 ({reduced_sample/sample_size})')

# Save the figure
plt.savefig('histograms/3DPockets.png', bbox_inches='tight')
plt.show()
