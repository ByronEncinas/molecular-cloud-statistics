from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
import seaborn as sns
import numpy as np
import copy

import json

# Assuming your JSON file is named 'data.json'
import glob

# Get a list of all files that match the pattern
file_list = glob.glob('random_distributed_reduction_factor*.json')

print(file_list)

for file_path in file_list:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        file_path = 'random_distributed_gas_density.json'
        # Load the JSON data into a Python list
        reduction_factor = json.load(file)

print(len(reduction_factor))

bins = len(reduction_factor)//10

# Assuming you have defined reduction_factor and bins already
counter = Counter(reduction_factor)
print(counter[1])

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
plt.savefig(f"histograms/hist={len(reduction_factor)}bins={bins}.png")

plt.show()

# Open the file in read mode
with open(file_path, 'r') as file:
    # Load the JSON data into a Python list
    gas_density = json.load(file)

dic_gas_r = {}
for gas, R in zip(gas_density, reduction_factor):
    dic_gas_r[gas] = R

ordered_dict_gas_r = OrderedDict(sorted(dic_gas_r.items()))
del gas_density, dic_gas_r

gas_density = list(ordered_dict_gas_r.keys())
reduction_f = list(ordered_dict_gas_r.values() )

from scipy import stats

# Extract data from the dictionary
x = np.log10(np.array(gas_density))   # log10(gas number density)
y = np.array(reduction_f)              # reduction factor R

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

plt.savefig(f"histograms/mean_median.png")

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
plt.savefig(f"histograms/mirrored_histograms.png")

# Show the plot
#plt.show()