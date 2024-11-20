import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import numpy as np
import json
import glob
import sys

snap = sys.argv[1]
norm = bool(sys.argv[2])

# Get a list of all files that match the pattern
#file_list = glob.glob('jsonfiles/random_distributed_reduction_factor*NO_IDEAL_300.json')
file_list = glob.glob(f'cluster_outputs/histCAS/1000/ideal_mhd_snap_{snap}/random_distributed_reduction_factorNO_ID.json')

print(file_list)
for file in file_list:
    print(file)

reduction_factor = []
for file_path in file_list:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data into a Python list and append to reduction_factor
        reduction_factor += list(json.load(file))

# Now reduction_factor contains the contents of all matching JSON files

# Get a list of all files that match the pattern
#file_list = glob.glob('jsonfiles/random_distributed_numb_density*NO_IDEAL_300.json')
file_list = glob.glob(f'cluster_outputs/histCAS/1000/ideal_mhd_snap_{snap}/random_distributed_numb_densityNO_ID.json')

numb_density = []
for file_path in file_list:
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data into a Python list and append to reduction_factor
        numb_density += list(json.load(file))

from collections import Counter

print("how many elements? (R) ",len(reduction_factor))
print("how many elements? (n) ",len(numb_density))
counter = Counter(reduction_factor)

print("how many zeroes? (R) ",counter[0])

counter = Counter(numb_density)
print("how many zeroes? (n)",counter[0])
numb_density = np.array(numb_density)
reduction_factor = np.array(reduction_factor)
#below = numb_density < 100
#numb_density[below]
#reduction_factor[below] = 1.0
#print(len(numb_density[below]))

bins = len(reduction_factor)//10
reduction_factor

# Update both reduction_factor and numb_density arrays
#reduction_factor = replace_zeros_with_half_of_second_min(reduction_factor)
#numb_density = replace_zeros_with_half_of_second_min(numb_density)

inverse_reduction_factor = [1/reduction_factor[i] for i in range(len(reduction_factor))]

bins = len(reduction_factor)//10

inverse_reduction_factor = [1/reduction_factor[i] for i in range(len(reduction_factor))]
counter = Counter(inverse_reduction_factor)

# Create a figure and axes objects
fig, axs = plt.subplots(1, 2, figsize=(9, 3))

# Plot histograms on the respective axes
axs[0].hist(reduction_factor, bins=bins, color='black',density=norm)
axs[0].set_yscale('log')
axs[0].set_title('Histogram of Reduction Factor (R)')
axs[0].set_xlabel('Bins')
axs[0].set_ylabel('Frequency')

axs[1].hist(inverse_reduction_factor, bins=bins, color='black',density=norm)
axs[1].set_yscale('log')
axs[1].set_title('Histogram of Inverse Reduction Factor (1/R)')
axs[1].set_xlabel('Bins')
axs[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f"hist={len(reduction_factor)}bins={bins}.png")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Assuming numb_density and reduction_factor are defined previously in your code.
# Extract data from the dictionary
x = np.log10(np.array(numb_density))   # log10(gas number density)
y = np.array(reduction_factor)          # reduction factor R

# Compute binned statistics
bins = len(numb_density) // 10

# Median binned statistics
bin_medians, bin_edges, _ = stats.binned_statistic(x, y, statistic='median', bins=bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Mean binned statistics
bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=bins)

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
plt.savefig(f"mean_median_mosaic.png")

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
bin_medians, bin_edges, _ = stats.binned_statistic(x, y, statistic='median', bins=bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Mean binned statistics
bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=bins)

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
# Set x-axis limits from 2 forward
axs[0].set_xlim(left=2)
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
plt.savefig(f"mean_median_scatter_mosaic.png")

# Show the plot
plt.show()


reduction_data = reduction_factor.copy()
density_data = numb_density.copy()
log_density_data = np.log10(density_data)

def stats(n):
    sample_r = []
    for i in range(0, len(density_data)):
        if np.abs(np.log10(density_data[i]/n)) < 1:
            sample_r.append(reduction_data[i])
    sample_r.sort()
    if len(sample_r) == 0:
        mean = None
        median = None
        ten = None
    else:
        mean = sum(sample_r)/len(sample_r)

        median = np.quantile(sample_r, .5)
        ten = np.quantile(sample_r, .1)

    return [mean, median, ten]

Npoints = len(reduction_factor)
x_n = np.logspace(2, 6, Npoints)
mean_vec = np.zeros(Npoints)
median_vec = np.zeros(Npoints)
ten_vec = np.zeros(Npoints)
for i in range(0, Npoints):
    s = stats(x_n[i])
    mean_vec[i] = s[0]
    median_vec[i] = s[1]
    ten_vec[i] = s[2]

num_bins = Npoints//100  # Define the number of bins as a variable

rdcut = []
for i in range(0, Npoints):
    if density_data[i] > 100:
        rdcut = rdcut + [reduction_data[i]]

fig = plt.figure(figsize = (12, 6))
ax1 = fig.add_subplot(121)
ax1.hist(rdcut, num_bins)  # Use the num_bins variable here
ax1.set_xlabel('Reduction factor', fontsize = 20)
ax1.set_ylabel('number', fontsize = 20)
plt.setp(ax1.get_xticklabels(), fontsize = 16)
plt.setp(ax1.get_yticklabels(), fontsize = 16)

ax2 = fig.add_subplot(122)
l1, = ax2.plot(x_n, mean_vec)
l2, = ax2.plot(x_n, median_vec)
l3, = ax2.plot(x_n, ten_vec)
plt.legend((l1, l2, l3), ('mean', 'median', '10$^{\\rm th}$ percentile'), loc = "lower right", prop = {'size':14.0}, ncol =1, numpoints = 5, handlelength = 3.5)
plt.xscale('log')
plt.ylim(0.25, 1.05)
ax2.set_ylabel('Reduction factor', fontsize = 20)
ax2.set_xlabel('gas density (hydrogens per cm$^3$)', fontsize = 20)
plt.setp(ax2.get_xticklabels(), fontsize = 16)
plt.setp(ax2.get_yticklabels(), fontsize = 16)

fig.subplots_adjust(left = .1)
fig.subplots_adjust(bottom = .15)
fig.subplots_adjust(top = .98)
fig.subplots_adjust(right = .98)

# Save the figure
plt.savefig('pocket_statistics_bjev.png')
plt.show()
# plot 3D cavities in CR density

if False:
    # Specify the file path
    file_path = f'position_vector_reduction'

    # Write the list data to a JSON file
    with open(file_path, 'w') as json_file:
        pos_red = json.load(json_file) # [x,y,z] = R basicly a 3D stochastic function

    radius_vector = []
    red_factor = []

    for k,v in pos_red.items():
        if v == 1:
            # we only want to plot where the R < 1
            continue
        radius_vector.append(k)
        red_factor.append(v)

    radius_vector = np.array(radius_vector)/ 3.086e+18  
    red_factor = np.array(red_factor)

    from mpl_toolkits.mplot3d import Axes3D

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = radius_vector[:,0]
    y = radius_vector[:,1]
    z = radius_vector[:,2]

    # Scatter plot with colormap based on the distance from origin
    sc = ax.scatter(x, y, z, c=red_factor, cmap='plasma')

    # Add colorbar to show the scale
    plt.colorbar(sc, label='Reduction Factor')            
    ax.set_xlabel('x [Pc]')
    ax.set_ylabel('y [Pc]')
    ax.set_zlabel('z [Pc]')
    ax.set_title('Pockets')

    plt.savefig(f'histograms/3DPockets.png', bbox_inches='tight') 
