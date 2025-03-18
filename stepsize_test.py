from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

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



fstepsizes = ['0.2', '0.4', '0.8']
rstepsizes = ['0.5', '1.0', '2.0']
colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Clear and accessible colors
markers = ['o', 'v', '^']  # Highly distinct markers
s = 5  # Larger marker size
alpha = 0.8  # Increase opacity

axes = ["$s$ (cm)", "$x - x_{cloud}$ (cm)","y - y_{cloud}$ (cm)","z - z_{cloud}$ (cm)"]
    
for name in axes: # with respect to which axis?

    for cycle in range(rounds): # how many files
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for idx, file in enumerate(fstepsizes): # how many stepsizes
            try:
                # Load data from files
                radius_vector = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArePositions{cycle}.npy", mmap_mode='r'))#/ 3.086e+18
                distance = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArepoTrajectory{cycle}.npy", mmap_mode='r'))#/ 3.086e+18
                bfield = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArepoMagneticFields{cycle}.npy", mmap_mode='r'))# *gauss_code_to_gauss_cgs
                numb_density = np.array(np.load(f"arepo_npys/stepsizetest/{file}/ArepoNumberDensities{cycle}.npy", mmap_mode='r'))
            except FileNotFoundError:
                continue

            # Print statements for debugging
            print(f"{file}")
            xcomp = radius_vector[:, 0]
            ycomp = radius_vector[:, 1]
            zcomp = radius_vector[:, 2]


            axes = {"$s$ (cm)":distance, "$x - x_{cloud}$ (cm)":xcomp, "y - y_{cloud}$ (cm)":ycomp, "z - z_{cloud}$ (cm)":zcomp}
            x_axis = axes[name]
            
            correction = float(file)
            fraction = round(float(file) / 0.4, 1)

            color = colors[idx]     # Assign color for current stepsize
            marker = markers[idx]   # Assign unique marker for current stepsize

            text = f"k={rstepsizes[idx]}"
            # Plot with scatter to use different markers
            axs[0].scatter(x_axis, bfield*1.0e+6, color=color, marker=marker, label=text,s=s,alpha=alpha)
            #axs[0].plot(x_axis, bfield*1.0e+6, color=color, label=text,alpha=alpha)
            axs[0].set_xlabel(name)
            axs[0].set_ylabel("Field strength ($\mu$G)")
            axs[0].set_title("$B$ field strength")
            axs[0].grid(True)

            axs[2].plot(distance, color=color, label=text)
            axs[2].set_xlabel("Steps")
            axs[2].set_yscale('log')
            axs[2].set_ylabel("$s$ (cm)")
            axs[2].set_title("Distance over path")
            axs[2].grid(True)

            #axs[1].plot(x_axis, numb_density, color=color, label=text,alpha=alpha)
            axs[1].scatter(x_axis, numb_density, color=color, marker=marker, label=text,s=s,alpha=alpha)
            axs[1].set_yscale('log')
            axs[1].set_xlabel(name)
            axs[1].set_ylabel("$n_g(s)$ cm$^{-3}$")
            axs[1].set_title("Number density")
            axs[1].grid(True)

        # Add legends for each subplot
        for ax in axs.flat:
            ax.legend()

        plt.tight_layout()

        # Save the figure
        plt.savefig(f"arepo_npys/stepsizetest/{name}-shapes_{cycle}.jpeg")

        # Close the plot to free memory
        plt.close(fig)