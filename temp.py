from collections import Counter, OrderedDict, defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import random
import time
import json
import sys
import os

# Assuming your JSON file is named 'data.json'
import glob

from library import *

start_time = time.time()

# read npy files and obtained bias version of the statistics using many random values in already obtained lines.

cluster = ("JAKAR", "CAS")

rounds = int(sys.argv[1])

import os

def list_files(directory, ext):
    import os
    # List all files in the specified directory
    all_files = os.listdir(directory)
    # Filter out only .npy files
    files = [f for f in all_files if f.endswith(f'.{ext}')]
    return files

# Example usage
directory_path = './arepo_npys/'
npy_files = list_files(directory_path, "hdf5")

reduction_factor = []
numb_density_at  = []
reduction_factor_at_gas_density = defaultdict()

import matplotlib.pyplot as plt
import numpy as np

stepsizes = ['0.8', '0.4', '0.2']  # List of stepsizes
colors = ['m', 'b', 'g', 'r']      # Define a color for each stepsize
markers = ['o', 's', '^', 'D']     # Define a unique marker for each stepsize

for cycle in range(rounds):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for idx, file in enumerate(stepsizes):
        try:
            # Load data from files
            radius_vector = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArePositions{cycle}.npy", mmap_mode='r'))
            distance = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArepoTrajectory{cycle}.npy", mmap_mode='r'))
            bfield = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArepoMagneticFields{cycle}.npy", mmap_mode='r'))
            numb_density = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArepoNumberDensities{cycle}.npy", mmap_mode='r'))
        except FileNotFoundError:
            continue

        # Print statements for debugging
        print(f"{file}")
        print(radius_vector[0, :] / 3.086e+18, distance[0] / 3.086e+18)
        print(radius_vector[-1, :] / 3.086e+18, distance[-1] / 3.086e+18)
        
        print(distance.shape, radius_vector.shape)
        
        correction = float(file)
        fraction = round(float(file) / 0.4, 1)

        color = colors[idx]     # Assign color for current stepsize
        marker = markers[idx]   # Assign unique marker for current stepsize

        # Plot with scatter to use different markers
        axs[0, 0].scatter(distance, bfield, color=color, marker=marker, label=f"Step {fraction}",s=1)
        axs[0, 0].set_xlabel("s (cm)")
        axs[0, 0].set_ylabel("$B(s)$ $\mu$G (cgs)")
        axs[0, 0].set_title("Magnetic Field")
        axs[0, 0].grid(True)

        axs[0, 1].plot(distance, color=color, label=f"Step {fraction}")
        axs[0, 1].set_xlabel("s (cm)")
        axs[0, 1].set_ylabel("$r$ cm (cgs)")
        axs[0, 1].set_title("Distance Away from MaxDensityCoord $r$")
        axs[0, 1].grid(True)

        axs[1, 0].scatter(distance, numb_density, color=color, marker=marker, label=f"Step {fraction}",s=1)
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_xlabel("s (cm)")
        axs[1, 0].set_ylabel("$N_g(s)$ cm^-3 (cgs)")
        axs[1, 0].set_title("Number Density (Nucleons/cm^3)")
        axs[1, 0].grid(True)

        axs[1, 1].scatter(distance, numb_density, color=color, marker=marker, label=f"Step {fraction}",s=1)
        axs[1, 1].set_yscale('log')
        axs[1, 1].set_xlabel("s (cm)")
        axs[1, 1].set_ylabel("$N_g(s)$ cm^-3 (cgs)")
        axs[1, 1].set_title("Number Density (Nucleons/cm^3)")
        axs[1, 1].grid(True)

    # Add legends for each subplot
    for ax in axs.flat:
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"arepo_npys/stepsizetest/shapes_{cycle}.jpeg")

    # Close the plot to free memory
    plt.close(fig)