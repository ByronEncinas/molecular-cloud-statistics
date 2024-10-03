from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import seaborn as sns
from scipy import stats
import numpy as np
import copy

import json

# Assuming your JSON file is named 'data.json'
import glob

# Get a list of all files that match the pattern
#file_list = glob.glob('jsonfiles/random_distributed_reduction_factor*.json')
file_list = glob.glob('random_distributed_reduction_factor*.json')

reduction_factor = []
for file_path in file_list:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data into a Python list and append to reduction_factor
        reduction_factor += list(json.load(file))

# Now reduction_factor contains the contents of all matching JSON files

# Get a list of all files that match the pattern
#file_list = glob.glob('jsonfiles/random_distributed_numb_density*.json')
file_list = glob.glob('random_distributed_numb_density*.json')

numb_density = []
for file_path in file_list:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data into a Python list and append to reduction_factor
        numb_density += list(json.load(file))

import itertools
from collections import Counter

print("how many elements?  ",len(reduction_factor))
print("how many elements?  ",len(numb_density))
counter = Counter(reduction_factor)

print("how many zeroes?  ",counter[0], counter[1])
print("is first element? ",reduction_factor[0])

counter = Counter(numb_density)
print("how many zeroes?  ",counter[0])
print("is first element? ",numb_density[0])

bins = len(reduction_factor)//10
reduction_factor

def replace_zeros_with_half_of_second_min(arr):
    # Convert to a NumPy array if it's not already
    arr = np.asarray(arr)

    # Step 1: Find the indices of all zero values
    zero_indices = np.where(arr == 0.0)[0]

    if len(zero_indices) == 0:  # If there are no zeros, return the array as is
        return arr

    # Step 2: Replace the zero values with infinity to ignore them in finding the second minimum
    arr[zero_indices] = np.inf

    # Step 3: Find the second minimum value
    second_min_value = np.min(arr)

    # Step 4: Replace all original zero values with half of the second minimum value
    arr[zero_indices] = second_min_value / 2

    return arr

# Update both reduction_factor and numb_density arrays
reduction_factor = replace_zeros_with_half_of_second_min(reduction_factor)
numb_density = replace_zeros_with_half_of_second_min(numb_density)

inverse_reduction_factor = [1/reduction_factor[i] for i in range(len(reduction_factor))]

bins = len(reduction_factor)//10

inverse_reduction_factor = [1/reduction_factor[i] for i in range(len(reduction_factor))]
counter = Counter(inverse_reduction_factor)

# Create a figure and axes objects
fig, axs = plt.subplots(1, 2, figsize=(9, 3))

# Plot histograms on the respective axes
axs[0].hist(reduction_factor, bins=bins, color='black')
axs[0].set_yscale('log')
axs[0].set_title('Histogram of Reduction Factor (R)')
axs[0].set_xlabel('Bins')
axs[0].set_ylabel('Frequency')

axs[1].hist(inverse_reduction_factor, bins=bins, color='black')
axs[1].set_yscale('log')
axs[1].set_title('Histogram of Inverse Reduction Factor (1/R)')
axs[1].set_xlabel('Bins')
axs[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f"histograms/hist={len(reduction_factor)}bins={bins}.png")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming numb_density and reduction_factor are defined previously in your code.
# Extract data from the dictionary
x = np.log10(np.array(numb_density))   # log10(gas number density)
y = np.array(reduction_factor)          # reduction factor R

# Compute binned statistics
num_bins = len(numb_density) // 50

# Median binned statistics
bin_medians, bin_edges, _ = stats.binned_statistic(x, y, statistic='median', bins=num_bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Mean binned statistics
bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)

# Overall mean and median
overall_mean = np.mean(y)
overall_median = np.median(y)

# Create a figure with two subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot bars for mean
axs[0].bar(bin_centers, bin_means, width=np.diff(bin_edges), color='salmon', alpha=0.6, edgecolor='red')
axs[0].axhline(overall_mean, color='blue', linestyle='--', label=f'Overall mean ($\\bar{{R}}$ = {overall_mean:.2f})')
axs[0].set_title('Binned Means')
axs[0].set_ylabel('$R$')
axs[0].set_xlabel('$\log_{10}(n_g \, [N/cm^{-3}])$')
axs[0].legend()
axs[0].grid()

# Plot bars for median
axs[1].bar(bin_centers, bin_medians, width=np.diff(bin_edges), color='lightblue', alpha=0.6, edgecolor='blue')
axs[1].axhline(overall_median, color='orange', linestyle='--', label=f'Overall median ($m_{{R}}$ = {overall_median:.2f})')
axs[1].set_title('Binned Medians')
axs[1].set_ylabel('$R$')
axs[1].set_xlabel('$\log_{10}(n_g \, [N/cm^{-3}])$')
axs[1].legend()
axs[1].grid()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f"histograms/mean_median_mosaic.png")

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming numb_density and reduction_factor are defined previously in your code.
# Extract data from the dictionary
x = np.log10(np.array(numb_density))   # log10(gas number density)
y = np.array(reduction_factor)          # reduction factor R

# Median binned statistics
bin_medians, bin_edges, _ = stats.binned_statistic(x, y, statistic='median', bins=num_bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Mean binned statistics
bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)

# Overall mean and median
overall_mean = np.mean(y)
overall_median = np.median(y)

# Create a figure with two subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot original scatter plot in the first subplot
axs[0].scatter(x, y, marker="x", s=1, color='red', label='Data points')
axs[0].set_title('Reduction Factor vs Log Density')
axs[0].set_ylabel('$R$')
axs[0].set_xlabel('$\log_{10}(n_g \, [N/cm^{-3}])$')
axs[0].grid()

# Plot bars for mean in the second subplot
axs[1].bar(bin_centers, bin_means, width=np.diff(bin_edges), color='salmon', alpha=0.6, edgecolor='red', label='Binned Mean')
axs[1].axhline(overall_mean, color='blue', linestyle='--', label=f'Overall mean ($\\bar{{R}}$ = {overall_mean:.2f})')
axs[1].set_title('Binned Means')
axs[1].set_ylabel('$R$')
axs[1].set_xlabel('$\log_{10}(n_g \, [N/cm^{-3}])$')
axs[1].legend()
axs[1].grid()

# Plot bars for median in the same subplot
axs[1].bar(bin_centers, bin_medians, width=np.diff(bin_edges), color='lightblue', alpha=0.4, edgecolor='blue', label='Binned Median')
axs[1].axhline(overall_median, color='orange', linestyle='--', label=f'Overall median ($m_{{R}}$ = {overall_median:.2f})')

# Add legend for the second subplot
axs[1].legend()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f"histograms/mean_median_scatter_mosaic.png")

# Show the plot
plt.show()
