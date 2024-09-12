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

max_cycles = int(sys.argv[1])
rounds = int(sys.argv[2])

import os

def list_npy_files(directory):
    # List all files in the specified directory
    all_files = os.listdir(directory)
    # Filter out only .npy files
    npy_files = [f for f in all_files if f.endswith('.npy')]
    return npy_files

# Example usage
directory_path = './arepo_npys/'
npy_files = list_npy_files(directory_path)

reduction_factor = []
numb_density_at  = []
reduction_factor_at_gas_density = defaultdict()

for cycle in range(max_cycles):
        
    for round in range(rounds):

        radius_vector  = np.array(np.load(f"arepo_npys/ArePositions{cycle}.npy", mmap_mode='r'))
        distance       = np.array(np.load(f"arepo_npys/ArepoTrajectory{cycle}.npy", mmap_mode='r'))
        bfield         = np.array(np.load(f"arepo_npys/ArepoMagneticFields{cycle}.npy", mmap_mode='r'))
        numb_density   = np.array(np.load(f"arepo_npys/ArepoNumberDensities{cycle}.npy", mmap_mode='r'))

        p_r = random.randint(0, len(distance) - 1)

        x_init = distance[p_r]
        B_init   = bfield[p_r]
        n_init = numb_density[p_r]

        #index_peaks, global_info = pocket_finder(bfield) # this plots
        pocket, global_info = pocket_finder(bfield, cycle, plot=False) # this plots
        index_pocket, field_pocket = pocket[0], pocket[1]

        # we can evaluate reduction factor if there are no pockets
        if len(index_pocket) < 2:
            # it there a minimum value of peaks we can work with? yes, two
            continue

        globalmax_index = global_info[0]
        globalmax_field = global_info[1]

        # Calculate the range within the 80th percentile
        start_index = len(bfield) // 10  # Skip the first 10% of indices
        end_index = len(bfield) - start_index  # Skip the last 10% of indices

        # we gotta find peaks in the interval   (B_l < random_element < B_h)
        # Generate a random index within the range
        #s_r = distance[p_r]
        B_r = bfield[p_r]

        print("random index: ", p_r, "peak's index: ", index_pocket)
        
        """How to find index of Bl?"""

        # Bl it is definitely between two peaks, we need to verify is also inside a pocket
        # such that Bl < Bs < Bh (p_i < p_r < p_j)

        # finds index at which to insert p_r and be kept sorted
        p_i = find_insertion_point(index_pocket, p_r)

        #print()
        print("Random Index:", p_r, "assoc. B(s_r):",B_r)
        print("Maxima Values related to pockets: ",len(index_pocket), p_i)

        if p_i is not None:
            # If p_i is not None, select the values at indices p_i-1 and p_i
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
        else:
            # If p_i is None, select the two closest values based on some other criteria
            continue

        if len(closest_values) == 2:
            B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
            B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
        else:
            R = 1
            reduction_factor.append(1)
            numb_density_at.append(n_init) 
            continue

        if B_r/B_l < 1:
            R = 1 - np.sqrt(1-B_r/B_l)
            reduction_factor.append(R)
            numb_density_at.append(n_init)
            continue
        else:
            R = 1
            reduction_factor.append(1)
            numb_density_at.append(n_init)
            continue
        
        print("Closest local maxima 'p':", closest_values)
        print("Bs: ", bfield[p_r], "Bi: ", bfield[closest_values[0]], "Bj: ", bfield[closest_values[1]])
        try:
            print("Bl: ", B_l, " B_r/B_l =", B_r/B_l, "< 1 ") 
        except:
            # this statement won't reach cycle += 1 so the cycle will continue again.
            print("Bl: ", B_l, " B_r/B_l =", B_r/B_l, "> 1 so CRs are not affected => R = 1") 

        # Now we pair reduction factors at one position with the gas density there.
        #gas_density_at_random = interpolate_scalar_field(point_i,point_j,point_k, gas_den)
        reduction_factor_at_gas_density[R] = numb_density_at # Key: 1/R => Value: Ng (gas density)

# Specify the file path
file_path = f'arepo_bias/random_distributed_reduction_factor{sys.argv[-1]}.json'

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(reduction_factor, json_file)

# Specify the file path
file_path = f'arepo_bias/random_distributed_gas_density{sys.argv[-1]}.json'

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(numb_density_at, json_file)

print(len(reduction_factor))

bins = len(reduction_factor)//10

# Assuming you have defined reduction_factor and bins already
counter = Counter(reduction_factor)

inverse_reduction_factor = [1/reduction_factor[i] for i in range(len(reduction_factor))]
print(len(inverse_reduction_factor))

# Create a figure and axes objects
fig, axs = plt.subplots(1, 2, figsize=(9, 3))

axs[0].hist(reduction_factor, bins=bins, color='skyblue')
axs[0].set_yscale('log')
axs[0].set_title('Histogram of Reduction Factor (R)')
axs[0].set_ylabel('Bins')
axs[0].set_xlabel('$R$')

control = np.ones_like(reduction_factor)

axs[1].hist(inverse_reduction_factor, bins=bins, color='skyblue')
axs[1].set_yscale('log')
axs[1].set_title('Histogram of Reduction Factor ($1/R$)')
axs[1].set_ylabel('Bins')
axs[1].set_xlabel('$log_{10}(1/R)$')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f"arepo_bias/hist={len(reduction_factor)}bins={bins}.png")

plt.show()

dic_gas_r = {}
for gas, R in zip(numb_density_at, reduction_factor):
    dic_gas_r[gas] = R

ordered_dict_gas_r = OrderedDict(sorted(dic_gas_r.items()))
del dic_gas_r

if True:

    # Extract data from the dictionary
    x = np.log10(np.array(numb_density_at))   # log10(gas number density)
    y = np.array(reduction_factor)              # reduction factor R

    # Plot original scatter plot
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    axs.scatter(x, y, marker="|", s=5, color='red', label='Data points')
    axs.set_title('Histogram of Reduction Factor (R)')
    axs.set_ylabel('$(R)$')
    axs.set_xlabel('$log_{10}(n_g ($N/cm^{-3}$))$ ')

    # Compute binned statistics
    num_bins = 100

    # Median binned statistics
    bin_medians, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='median', bins=num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs.plot(bin_centers, bin_medians, marker="+", color='#17becf', linestyle='-', label='Binned medians')

    # Mean binned statistics
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)
    axs.plot(bin_centers, bin_means, marker="x", color='pink', linestyle='-', label='Binned means')

    # Overall mean and median
    overall_mean = np.average(y)
    overall_median = np.median(y)

    mean = np.ones_like(y) * overall_mean
    median = np.ones_like(y) * overall_median

    axs.plot(x, mean, color='dimgrey', linestyle='--', label=f'Overall mean ({overall_mean:.2f})')
    axs.plot(x, median, color='dimgray', linestyle='--', label=f'Overall median ({overall_median:.2f})')

    # Add legend
    axs.legend()

    plt.savefig(f"arepo_bias/mean_median.png")
    plt.close(fig)
    #plt.show()

    # Define the number of bins
    num_bins = 100

    # Compute binned statistics
    bin_medians, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='median', bins=num_bins)
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create the figure and axis
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    # Plot the histograms using Matplotlib
    axs.hist(bin_edges[:-1], bins=bin_edges, weights=bin_medians, alpha=0.5, label='medians', color='c', edgecolor='darkcyan')
    axs.hist(bin_edges[:-1], bins=bin_edges, weights=-bin_means, alpha=0.5, label='means', color='m', edgecolor='darkmagenta')

    # Set the labels and title
    axs.set_title('Histograms of Binned Medians and Means (Inverted)')
    axs.set_ylabel('$(R)$')
    axs.set_xlabel('$log_{10}(n_g ($N/cm^{-3}$))$ ')

    # Add legend
    axs.legend(loc='center')

    # save figure
    plt.savefig(f"arepo_bias/mirrored_histograms.png")

    # Show the plot
    plt.close(fig)
    #plt.show()