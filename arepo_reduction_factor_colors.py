from collections import defaultdict
from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
from scipy import spatial
import healpy as hp
import numpy as np
import random
import glob
import os
import h5py
import json
import sys
from library import *

import time

start_time = time.time()


"""  
Using Margo Data

Analysis of reduction factor

$$N(s) 1 - \sqrt{1-B(s)/B_l}$$

Where $B_l$ corresponds with (in region of randomly choosen point) the lowest between the highest peak at both left and right.
where $s$ is a random chosen point at original 256x256x256 grid.

1.- Randomly select a point in the 3D Grid. 
2.- Follow field lines until finding B_l, if non-existent then change point.
3.- Repeat 10k times
4.- Plot into a histogram.

contain results using at least 20 boxes that contain equally spaced intervals for the reduction factor.

# Calculating Histogram for Reduction Factor in Randomized Positions in the 128**3 Cube 

"""

"""
Parameters

- [N] default is 50 as the total number of steps in the simulation
- [dx] default 4/N of the rloc_boundary (radius of spherical region of interest) variable

"""
FloatType = np.float64
IntType = np.int32

if len(sys.argv)>4:
    #python3 arepo_reduction_factor_colors.py N rloc max_cycles num_file SLURM_JOB_ID
    #python3 arepo_reduction_factor_colors.py 500 0.5 100 430 
	N=int(sys.argv[1])
	rloc_boundary=float(sys.argv[2])
	max_cycles   =int(sys.argv[3])
	num_file = f'{sys.argv[4]}'
	if len(sys.argv) < 6:
		sys.argv.append('NO_ID')
else:
    N            =10_000
    rloc_boundary=1   # rloc_boundary for boundary region of the cloud
    max_cycles   =10
    num_file = '430'

print(*sys.argv)

# flow control to repeat calculations in no peak situations
cycle = 0 

reduction_factor_at_numb_density = defaultdict()

reduction_factor = []

"""  B. Jesus Velazquez """

file_list = glob.glob('arepo_data/*.hdf5')
print(file_list)

for f in file_list:
    if num_file in f:
        filename = f

print(filename)

data = h5py.File(filename, 'r')
# Access the 'Header' group

header_group = data['Header']

# Retrieve the 'Time' attribute from the 'Header' group
time_value = header_group.attrs['Time']

snap = filename.split('/')[1].split('.')[0]

new_folder = os.join("histograms/" , snap)

Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Volume   = Mass/Density

# Initialize gradients
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))

# Print the time
print(f"Time: {time_value} Myr")

# printing relevant info about the data
# 1 Parsec  = 3.086e+18 cm
# 1 Solar M = 1.9885e33 gr
# 1 Km/s    = 100000 cm/s

"""  
Attribute: UnitLength_in_cm = 3.086e+18
Attribute: UnitMass_in_g = 1.99e+33
Attribute: UnitVelocity_in_cm_per_s = 100000.0

Name: PartType0/Coordinates
    Attribute: to_cgs = 3.086e+18
Name: PartType0/Density
    Attribute: to_cgs = 6.771194847794873e-23
Name: PartType0/Masses
    Attribute: to_cgs = 1.99e+33
Name: PartType0/Velocities
    Attribute: to_cgs = 100_000.0
Name: PartType0/MagneticField

"""

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

#Center= 0.5 * Boxsize * np.ones(3) # Center
#Center = np.array( [91,       -110,          -64.5]) #117
#Center = np.array( [96.6062303,140.98704002, 195.78020632]) #117
Center = Pos[np.argmax(Density),:] #430

# all positions are relative to the 'Center'
VoronoiPos-=Center
Pos-=Center

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

def get_along_lines(x_init):

    dx = 1.0

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
        
        #print(threshold)
        #
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
                with open(f'lone_run_radius_vectors{snap}.dat', 'a') as file: 
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
                with open(f'lone_run_radius_vectors{snap}.dat', 'a') as file: 
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
    trajectory      = np.zeros_like(magnetic_fields)
    radius_to_origin= np.zeros_like(magnetic_fields)

    print("Magnetic fields shape:", magnetic_fields.shape)
    print("Radius vector shape:", radius_vector.shape)
    print("Numb densities shape:", numb_densities.shape)

    m = magnetic_fields.shape[1]
    
    print("Survinving lines: ", m, "out of: ", max_cycles)
	
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

rloc_center      = np.array([float(random.uniform(0,rloc_boundary)) for l in range(max_cycles)])
nside = 1_000     # sets number of cells sampling the spherical boundary layers = 12*nside**2
npix  = 12 * nside ** 2
ipix_center       = np.arange(npix)
xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix_center)

xx = np.array(random.sample(list(xx), max_cycles))
yy = np.array(random.sample(list(yy), max_cycles))
zz = np.array(random.sample(list(zz), max_cycles))

m = len(zz) # amount of values that hold which_up_down

print("Values of reduction factor to be generated: ",m)

x_init = np.zeros((m,3))

x_init[:,0]      = rloc_center * xx
x_init[:,1]      = rloc_center * yy
x_init[:,2]      = rloc_center * zz

print("Cores Used          : ", os.cpu_count())
print("Steps in Simulation : ", 2*N)
print("rloc_boundary       : ", rloc_boundary)
print("rloc_center         : ", rloc_center)
print("max_cycles          : ", max_cycles)
print("Boxsize             : ", Boxsize) # 256
print("Center              : ", Center) # 256
print("Posit Max Density   : ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume     : ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume     : ", Volume[np.argmax(Volume)]) # 256
print(f"Smallest Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmax(Volume)]}")
print(f"Biggest  Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmin(Volume)]}")

__, radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th = get_along_lines(x_init)

print("Elapsed Time: ", (time.time() - start_time)/60.)

with open(os.join(new_folder, 'output'), 'w') as file:
    file.write(f"{filename}\n")
    file.write(f"Cores Used: {os.cpu_count()}\n")
    file.write(f"Snap Time (Myr): {time_value}\n")
    file.write(f"rloc_boundary (Pc) : {rloc_boundary}\n")
    file.write(f"rloc_center (Pc)   :\n {rloc_center}\n")
    file.write(f"x_init (Pc)        :\n {x_init}\n")
    file.write(f"max_cycles         : {max_cycles}\n")
    file.write(f"Boxsize (Pc)       : {Boxsize} Pc\n")
    file.write(f"Center (Pc, Pc, Pc): {Center[0]}, {Center[1]}, {Center[2]} \n")
    file.write(f"Posit Max Density (Pc, Pc, Pc): {Pos[np.argmax(Density), :]}\n")
    file.write(f"Smallest Volume (Pc^3)   : {Volume[np.argmin(Volume)]} \n")
    file.write(f"Biggest  Volume (Pc^3)   : {Volume[np.argmax(Volume)]}\n")
    file.write(f"Smallest Density (M☉/Pc^3)  : {Density[np.argmax(Volume)]} \n")
    file.write(f"Biggest  Density (M☉/Pc^3) : {Density[np.argmin(Volume)]}\n")
    file.write(f"Smallest Density (N/cm^3)  : {Density[np.argmax(Volume)]*gr_cm3_to_nuclei_cm3} \n")
    file.write(f"Biggest  Density (N/cm^3) : {Density[np.argmin(Volume)]*gr_cm3_to_nuclei_cm3}\n")
    file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")

# flow control to repeat calculations in no peak situations

m = magnetic_fields.shape[1]

threshold, threshold_rev = th

reduction_factor = list()
numb_density_at  = list()

min_den_cycle = list()

pos_red = dict()

for cycle in range(max_cycles):

    _from = N+1 - threshold_rev[cycle]
    _to   = N+1 + threshold[cycle]
    #print(f"{_from} - {_to}")
    p_r = N +1 - _from

    bfield    = magnetic_fields[_from:_to,cycle]
    distance = trajectory[_from:_to,cycle]
    numb_density = numb_densities[_from:_to,cycle]
    tupi = f"{x_init[cycle,0]},{x_init[cycle,1]},{x_init[cycle,2]}"

    #index_peaks, global_info = pocket_finder(bfield) # this plots
    pocket, global_info = pocket_finder(bfield, cycle, plot=True) # this plots
    index_pocket, field_pocket = pocket[0], pocket[1]

    min_den_cycle.append(min(numb_density))
    
    globalmax_index = global_info[0]
    globalmax_field = global_info[1]

    x_r = distance[p_r]
    B_r = bfield[p_r]
    n_r = numb_density[p_r]

    # finds index at which to insert p_r and be kept sorted
    p_i = find_insertion_point(index_pocket, p_r)

    print("random index: ", p_r, "assoc. B(s_r), n_g(s_r):",B_r, n_r, "peak's index: ", index_pocket)
    
    """How to find index of Bl?"""

    print("Maxima Values related to pockets: ", len(index_pocket), p_i)
    try:
        # possible error is len(index_pocket) is only one or two elements
        closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
        B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
        B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
        success = True  # Flag to track if try was successful

    except:
        R = 1
        reduction_factor.append(R)
        numb_density_at.append(n_r)
        pos_red[tupi] = R
        success = False  # Set flag to False if there's an exception
        continue

    # Only execute this block if try was successful
    if success:
        if B_r / B_l < 1:
            R = 1 - np.sqrt(1 - B_r / B_l)
            reduction_factor.append(R)
            numb_density_at.append(n_r)
            pos_red[tupi] = R
        else:
            R = 1
            reduction_factor.append(R)
            numb_density_at.append(n_r)
            pos_red[tupi] = R

    print("Closest local maxima 'p':", closest_values)
    print("Bs: ", B_r, "ns: ", n_r)
    print("Bi: ", bfield[closest_values[0]], "Bj: ", bfield[closest_values[1]])

    if B_r/B_l < 1:
        print(" B_r/B_l =", B_r/B_l, "< 1 ") 
    else:
        # this statement won't reach cycle += 1 so the cycle will continue again.
        print(" B_r/B_l =", B_r/B_l, "> 1 so CRs are not affected => R = 1") 

from collections import Counter

#print(reduction_factor)
#print(numb_density_at)
counter = Counter(reduction_factor)
#counter[0]

pos_red = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pos_red.items()}

# Create the new arepo_npys directory
os.makedirs(new_folder, exist_ok=True)

# Print elapsed time
print(f"Elapsed time: {(time.time() - start_time)/60.} Minutes")

# Specify the file path
file_path = os.join(new_folder, f'random_distributed_reduction_factor{sys.argv[-1]}.json')

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(reduction_factor, json_file)

# Specify the file path
file_path = os.join(new_folder,f'random_distributed_numb_density{sys.argv[-1]}.json')

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(numb_density_at, json_file)

# Specify the file path
file_path = os.join(new_folder,f'position_vector_reduction{sys.argv[-1]}')

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(pos_red, json_file) # [x,y,z] = R basicly a 3D stochastic functin

"""# Graphs"""

#plot_trajectory_versus_magnitude(trajectory, magnetic_fields, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])

bins=len(reduction_factor)//10 

if bins == 0:
    bins=1

inverse_reduction_factor = [1/reduction_factor[i] for i in range(len(reduction_factor))]

# try plt.stairs(*np.histogram(inverse_reduction_factor, 50), fill=True, color='skyblue')

# Create a figure and axes objects
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot histograms on the respective axes
axs[0].hist(reduction_factor, bins=bins, color='skyblue', edgecolor='black')
axs[0].set_yscale('log')
axs[0].set_title('Histogram of Reduction Factor (R)')
axs[0].set_xlabel('Bins')
axs[0].set_ylabel('Frequency')

axs[1].hist(inverse_reduction_factor, bins=bins, color='skyblue', edgecolor='black')
axs[1].set_yscale('log')
axs[1].set_title('Histogram of Inverse Reduction Factor (1/R)')
axs[1].set_xlabel('Bins')
axs[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Save the figure
#plt.savefig("c_output_data/histogramdata={len(reduction_factor)}bins={bins}"+name+".png")
plt.savefig(os.join(new_folder,f"hist={len(reduction_factor)}bins={bins}.png"))

# Show the plot
#plt.show()

from scipy import stats
import seaborn as sns

if True:

    # Extract data from the dictionary
    x = np.log10(numb_density_at)   # log10(numb number density)
    y = np.array(reduction_factor)              # reduction factor R

    # Plot original scatter plot
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    axs.scatter(x, y, marker="x", s=5, color='red', label='Data points')
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

    plt.savefig(os.join(new_folder,f"mean_median.png"))
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
    plt.savefig(os.join(new_folder,f"mirrored_histograms.png"))

    # Show the plot
    plt.close(fig)
    #plt.show()