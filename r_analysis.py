from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import sys, time, glob, re
import warnings, csv

pc_to_cm = 3.086 * 10e+18  # cm per parsec

start_time = time.time()

amb_bundle   = sorted(glob.glob('./thesis_stats/amb/*/DataBundle*.npz'))
ideal_bundle = sorted(glob.glob('./thesis_stats/ideal/*/DataBundle*.npz'))

#amb_bundle   = sorted(glob.glob('../../thesis_figures/thesis_stats/amb/*/DataBundle*.npz'))
#ideal_bundle = sorted(glob.glob('../../thesis_figures/thesis_stats/ideal/*/DataBundle*.npz'))

#print(ideal_bundle)

bundle_dirs = [ideal_bundle,amb_bundle]

readable = "{:02}:{:06.3f}".format(int((time.time()-start_time) // 60), (time.time()-start_time)  % 60)

def extract_number(filename):
    match = re.search(r'(\d+)(?=\_.json)', filename)
    return int(match.group(0)) if match else 0

def pocket_finder(bfield, r, B, img = '', plot=False):
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    try:
        idx = index_global_max[0]
    except:
        idx = index_global_max
    upline == bfield[idx]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001


    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]

    if plot:
        # pocket with density threshold 100cm-3
        mask = np.log10(numb)<2
        slicebelow = mask[:p_r]
        sliceabove = mask[p_r:]

        above100 = np.where(sliceabove == True)[0][0] + p_r
        below100 = np.where(slicebelow == True)[0][-1]    
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.plot(bfield)

        axs.vlines(below100, bfield[below100]*(1-0.1),bfield[below100]*(1+0.1),color='black', label='th 100cm-3 (left)')
        axs.vlines(above100, bfield[above100]*(1-0.1),bfield[above100]*(1+0.1),color='black', label='th 100cm-3 (right)')
        axs.plot(r, B, "s", color="black", alpha = 0.4)
        axs.plot(indexes, peaks, "x", color="green")

        axs.plot(indexes, peaks, ":", color="green")
        
        #for idx in index_global_max:
        axs.plot(idx, upline, "x", color="black")
        axs.plot(np.ones_like(bfield) * baseline, "--", color="gray")
        axs.set_xlabel("Index")
        axs.set_ylabel("Field")
        axs.set_title("Field Shape")
        axs.legend(["bfield", "all peaks", "index_global_max", "baseline"])
        axs.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure

        plt.savefig(f"./pockets/field_shape{img}.png")
        plt.savefig(f"./field_shape.png")
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def evaluate_reduction(field, numb, thresh):
    R10 = []
    R100 = []
    Numb100 = []
    RBundle = []
    m = field.shape[1]

    for i in range(m):

        shape = magnetic_fields.shape[0]

        # threshold, threshold2, threshold_rev, threshold2_rev
        x10, x100, y10, y100 = threshold[:,i]

        # to slice bfield with threshold 10cm-3
        xp10 = shape//2 + x10
        xm10 = shape//2 - y10

        try:
            numb   = numb_densities[xm10-1:xp10+1,i]
            bfield = magnetic_fields[xm10-1:xp10+1,i]
        except:
            raise ValueError("Trying to slice a list outisde of its boundaries")

        p_r = shape//2 - xm10-1
        B_r = bfield[p_r]
        n_r = numb[p_r]

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield, p_r, B_r, img=i, plot=False)
        index_pocket, field_pocket = pocket[0], pocket[1]

        p_i = np.searchsorted(index_pocket, p_r)
        
        # are there local maxima around our point? 
        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
            B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
            # YES! 
            success = True  
        except:
            # NO :c
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R10.append(R)
                Numb100.append(n_r)
            else:
                # NO!
                R = 1.
                R10.append(R)
                Numb100.append(n_r)
        del closest_values, success, B_l, B_h, R

        # pocket with density threshold 100cm-3
        mask = np.log10(numb)<2
        slicebelow = mask[:p_r]
        sliceabove = mask[p_r:]

        above100 = np.where(sliceabove == True)[0][0] + p_r
        below100 = np.where(slicebelow == True)[0][-1]    

        numb   = numb[below100-1:above100+1]
        bfield = bfield[below100-1:above100+1]

        # original size N, new size N'
        # cuts where done from 0 - below100 and above100 - N
        # p_r is the index of the generating point
        # what is p_r?
        
        p_r = p_r - below100 + 1

        #print(p_r, np.round(bfield[p_r], 4))
        
        B_r = bfield[p_r]

        pocket, global_info = pocket_finder(bfield, p_r, B_r, plot=False)
        index_pocket, field_pocket = pocket[0], pocket[1]

        p_i = np.searchsorted(index_pocket, p_r)

        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
            B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
            success = True  
        except:
            R = 1
            R100.append(R)

            success = False 
            continue
        if success:
            if B_r / B_l < 1:
                R = 1 - np.sqrt(1 - B_r / B_l)
                R100.append(R)

            else:
                R = 1
                R100.append(R)

        #print(R10[-1] - R100[-1])
        RBundle.append((R10, R100))

    return RBundle, R10, R100, Numb100

def statistics_reduction(R, N):
    # R is numpy array
    def _stats(n, d_data, r_data, p_data=0):
        sample_r = []

        for i in range(0, len(d_data)):
            if np.abs(np.log10(d_data[i]/n)) < 0.2:
                sample_r.append(r_data[i])
        sample_r.sort()
        if len(sample_r) == 0:
            mean = None
            median = None
            ten = None
            size = 0
        else:
            mean = sum(sample_r)/len(sample_r)
            median = np.quantile(sample_r, .5)
            ten = np.quantile(sample_r, .1)
            size = len(sample_r)
        return [mean, median, ten, size]

    total = len(R)
    ones  = np.sum(R==1)
    not_ones  = total - ones
    #print(ones, total)
    f = ones/total
    ncrit = 100
    mask = R<1
    R = R[mask]
    N = N[mask]
    minimum, maximum = np.min(np.log10(N)), np.max(np.log10(N))
    Npoints = len(R)

    x_n = np.logspace(2, maximum, Npoints)
    mean_vec = np.zeros(Npoints)
    median_vec = np.zeros(Npoints)
    ten_vec = np.zeros(Npoints)
    sample_size = np.zeros(Npoints)
    for i in range(0, Npoints):
        s = _stats(x_n[i], N, R)
        mean_vec[i] = s[0]
        median_vec[i] = s[1]
        ten_vec[i] = s[2]
        sample_size[i] = s[3]
    
    num_bins = Npoints//10  # Define the number of bins as a variable

    return R, x_n, mean_vec, median_vec, ten_vec, sample_size, f, N

R100 = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
R10  = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
NR   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
CD   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
Delta = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
delta = 0.0

peak_den    = OrderedDict({'ideal': [], 'amb': []})
snap_values = OrderedDict({'ideal': [], 'amb': []})
time_values = OrderedDict({'ideal': [], 'amb': []})


for bundle_dir in bundle_dirs: # ideal and ambipolar
    if bundle_dir == []:
        continue
    case = str(bundle_dir[0].split('/')[-3])
    repeated = set()
    for snap_data in bundle_dir: # from 000 to 490 
        snap = str(snap_data.split('/')[-2])

        
        # Path to the input file
        file_path = f'./{case}_cloud_trajectory.txt'

        # Regex pattern to match the line where the first column equals 'snap'
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if snap == str(row[0]) and snap not in repeated:

                    repeated.add(snap)
                    snap_values[case].append(str(row[0]))
                    time_values[case].append(float(row[1]))
                    peak_den[case].append(float(row[-1]))
                    readable = "{:02}:{:06.3f}".format(int((time.time()-start_time) // 60), (time.time()-start_time)  % 60)
                    break
        # Load lazily
        data = np.load(snap_data, mmap_mode='r')

        # Function to get size in MB
        def size_in_mb(array):
            return array.nbytes / 1e6  # Convert bytes to MB

        column_density = data['column_density']
        radius_vector = data['positions']
        trajectory = data['trajectory']
        numb_densities = data['number_densities']
        magnetic_fields = data['magnetic_fields']
        threshold = data['thresholds']

        # Print sizes
        #print("Memory usage:")
        #print(f"column_density: {size_in_mb(column_density):.2f} MB")
        #print(f"positions:       {size_in_mb(radius_vector):.2f} MB")
        #print(f"trajectory:      {size_in_mb(trajectory):.2f} MB")
        #print(f"number_densities:{size_in_mb(numb_densities):.2f} MB")
        #print(f"magnetic_fields: {size_in_mb(magnetic_fields):.2f} MB")
        #print(f"thresholds:      {size_in_mb(threshold):.2f} MB")
        """
                total = sum([
                    column_density.nbytes,
                    radius_vector.nbytes,
                    trajectory.nbytes,
                    numb_densities.nbytes,
                    magnetic_fields.nbytes,
                    threshold.nbytes
                ])
        if False:
            x = radius_vector[numb_densities.shape[0]//2,:,0]/pc_to_cm
            y = radius_vector[numb_densities.shape[0]//2,:,1]/pc_to_cm
            z = radius_vector[numb_densities.shape[0]//2,:,2]/pc_to_cm

            rloc = 0.1
            x = x[z<0.02 and z>-0.02]
            y = y[z<0.02 and z>-0.02]
            
            log_n = np.log10(numb_densities[numb_densities.shape[0]//2,:])


            plt.figure(figsize=(6, 5))
            sc = plt.scatter(x, y, c=log_n, cmap='viridis', s=10, edgecolor='none')
            plt.colorbar(sc, label=r'$\log_{10}(n)$')
            plt.xlabel('x [pc]')
            plt.ylabel('y [pc]')
            plt.title('XY Projection Colored by log$_{10}$(n)')
            plt.axis('equal')  # to keep scale of x and y consistent
            plt.tight_layout()
            plt.show()
        """

        #print(f"Elements: ", radius_vector.shape)
        #print(f"\nTotal: {total / 1e6:.2f} MB = {total / 1e9:.2f} GB")

        r_bundle, r_10, r_100, n_r = evaluate_reduction(magnetic_fields, numb_densities, threshold)

        readable = "{:02}:{:06.3f}".format(int((time.time()-start_time) // 60), (time.time()-start_time)  % 60)
        CD[case][snap]   =  CD[case].get(snap,  list(column_density[-1,:].tolist()*0)) + column_density[-1,:].tolist()
        R10[case][snap]  =  R10[case].get(snap,  list(r_10*0)) + list(r_10)
        R100[case][snap] = R100[case].get(snap, list(r_100*0)) + list(r_100)
        NR[case][snap] = NR[case].get(snap, list(n_r*0))+ list(n_r)



mean_ideal   = []
median_ideal = []
mean_amb     = []
median_amb   = []

ideal_snap  = []
amb_snap    = []

s_ideal = []
s_amb   = []

fractions_i = []
fractions_a = []

ideal_time = time_values['ideal']
amb_time = time_values['amb']
#print(ideal_time)

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

# --- IDEAL CASE ---
labels = list(CD['ideal'].keys())  
data = list(CD['ideal'].values())  

round_time = [np.round(t, 6) for t in ideal_time]

fig, ax = plt.subplots()
ax.boxplot(data, flierprops=dict(marker='|', markersize=2, color='red'))
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title('Column Density along CR path (ideal)')
ax.set_yscale('log')
ax.grid(True)
ax.set_xticklabels(round_time, rotation=60)
ax.locator_params(axis='x', nbins=15)
plt.tight_layout()
plt.savefig("./path_cd_ideal.png")
plt.close()

median = np.array([np.median(arr) for arr in data])
mean = np.array([np.mean(arr) for arr in data])
lower_68 = np.array([np.percentile(arr, 16) for arr in data])
upper_68 = np.array([np.percentile(arr, 84) for arr in data])
lower_95 = np.array([np.percentile(arr, 2.5) for arr in data])
upper_95 = np.array([np.percentile(arr, 97.5) for arr in data])

#print(median)

import numpy as np

# Create evenly spaced X indices
x_idx = np.arange(len(round_time))  # [0, 1, 2, ..., N-1]

fig, ax = plt.subplots()

# Use indices for plotting
ax.fill_between(x_idx, lower_95, upper_95, color='pink', label='Interpercentile range (2.5th–97.5th)')
ax.fill_between(x_idx, lower_68, upper_68, color='red', alpha=0.6, label='Interpercentile range (16th–84th)')
ax.plot(x_idx, median, color='black', linestyle='--', linewidth=1, label='Median')
ax.plot(x_idx, mean, color='black', linestyle='-', linewidth=1, label='Mean')

# Set tick positions and corresponding labels
ax.set_xticks(x_idx)
ax.set_xticklabels(round_time, rotation=60)

# Formatting
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title('Column Density along CR path (ideal)')
ax.set_yscale('log')
ax.grid(True)
ax.legend(loc='upper left', frameon=True, fontsize=16)

plt.tight_layout()
plt.savefig("./path_cd_ideal_inter.png")
plt.close()

# --- AMB CASE ---
labels = list(CD['amb'].keys())  # snapshot labels
data = list(CD['amb'].values())  # column density arrays

round_time = [np.round(t, 6) for t in amb_time]
fig, ax = plt.subplots()
ax.boxplot(data, flierprops=dict(marker='|', markersize=2, color='red'))
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title('Column Density along CR path (amb)')
ax.set_yscale('log')
ax.grid(True)
ax.set_xticklabels(round_time, rotation=60)
ax.locator_params(axis='x', nbins=15)
plt.tight_layout()
plt.savefig("./path_cd_amb.png")
plt.close()

median = np.array([np.median(arr) for arr in data])
mean = np.array([np.mean(arr) for arr in data])
lower_68 = np.array([np.percentile(arr, 16) for arr in data])
upper_68 = np.array([np.percentile(arr, 84) for arr in data])
lower_95 = np.array([np.percentile(arr, 2.5) for arr in data])
upper_95 = np.array([np.percentile(arr, 97.5) for arr in data])

import numpy as np

# Create evenly spaced X indices
x_idx = np.arange(len(round_time))  # [0, 1, 2, ..., N-1]

fig, ax = plt.subplots()

ax.fill_between(x_idx, lower_95, upper_95, color='pink', label='Interpercentile range (2.5th–97.5th)')
ax.fill_between(x_idx, lower_68, upper_68, color='red', alpha=0.6, label='Interpercentile range (16th–84th)')
ax.plot(x_idx, median, color='black', linestyle='--', linewidth=1, label='Median')
ax.plot(x_idx, mean, color='black', linestyle='-', linewidth=1, label='Mean')
ax.set_xticks(x_idx)
ax.set_xticklabels(round_time, rotation=60)
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title('Column Density along CR path (amb)')
#ax.set_yscale('log')
ax.grid(True)
ax.legend(loc='upper left', frameon=True, fontsize=16)
plt.savefig(f"./path_cd_amb_inter.png")
plt.close()

for k, v in CD['ideal'].items():
    #print("CD size: ",len(CD['ideal'][k]), np.max(CD['ideal'][k]))
    CD['ideal'][k]   =   np.mean(CD['ideal'][k])

for k, v in CD['amb'].items():
    #print("CD size: ",len(CD['amb'][k]), np.max(CD['amb'][k]))
    CD['amb'][k]   =   np.mean(CD['amb'][k])

for k, v in sorted(R10['ideal'].items()):
    mean_ideal.append(np.mean((np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])))
    median_ideal.append(np.median((np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])))
    ideal_snap.append(int(k))
    r_, x, mean, median, ten, s_size, f, n_ = statistics_reduction(np.array(R10['ideal'][k]), np.array(NR['ideal'][k])) 
    s_ideal.append((r_, x, mean, median, ten, s_size, k, f, n_))
    fractions_i.append(float(f))
    Delta['ideal'][k] = (np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])

for k, v in sorted(R10['amb'].items()):
    mean_amb.append(np.mean((np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])))
    median_amb.append(np.median((np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])))
    amb_snap.append(int(k))
    r_, x, mean, median, ten, s_size, f, n_ = statistics_reduction(np.array(R10['amb'][k]), np.array(NR['amb'][k])) 
    s_amb.append((r_, x, mean, median, ten, s_size, k, f, n_))
    fractions_a.append(float(f))
    Delta['amb'][k] = (np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])


fig0, ax0 = plt.subplots()
ax0.plot(ideal_time, mean_ideal , label='mean $\\Delta_{ideal}$', linewidth=1.5, linestyle='-', color='darkorange')
ax0.plot(ideal_time, median_ideal, label='median $\\Delta_{ideal}$', linewidth=1.5, linestyle='--', color='darkorange')
ax0.plot(amb_time, mean_amb, label='mean $\\Delta_{amb}$', linewidth=1.5, linestyle='-', color='royalblue')
ax0.plot(amb_time, median_amb, label='median $\\Delta_{amb}$', linewidth=1.5, linestyle='--', color='royalblue')
ax0.set_ylabel('$\\Delta = (R_{10} - R_{100})/R_{100}$')
#ax0.set_title('$\\Delta = (R_{10} - R_{100})/R_{100}$')
ax0.set_xlabel('Time (Myrs)')
ax0.legend(frameon=False)
plt.subplots_adjust(left=0.15)  # Increase left margin
plt.savefig('./delta_threshold.png')
plt.close()

mean_ir     = []
median_ir   = []
percen25_ir = []
percen10_ir = []

for s, r in R100['ideal'].items():

    array = np.array(r)
    r_ideal = array[array<1]
    mean_ir += [np.mean(r_ideal)]
    median_ir += [np.median(r_ideal)]
    percen25_ir += [np.percentile(r_ideal,25)]
    percen10_ir += [np.percentile(r_ideal,10)]
    #print(np.percentile(r_ideal,25), np.percentile(r_ideal,10), np.mean(r_ideal), np.median(r_ideal))

mean_ar     = []
median_ar   = []
percen25_ar = []
percen10_ar = []

for s, r in R100['amb'].items():
    array = np.array(r)
    r_amb = array[array<1]
    mean_ar += [np.mean(r_amb)]
    median_ar += [np.median(r_amb)]
    percen25_ar += [np.percentile(r_amb,25)]
    percen10_ar += [np.percentile(r_amb,10)]

fig_ideal, ax_ideal = plt.subplots()
fig_amb, ax_amb = plt.subplots()

ax_ideal.scatter(ideal_time, fractions_i, marker='x', color='black', s=8)
ax_ideal.plot(ideal_time, mean_ir, label='mean', linewidth=1.5, linestyle='-', color='royalblue')
ax_ideal.plot(ideal_time, median_ir, label='median', linewidth=1.5, linestyle='--', color='darkorange')
ax_ideal.plot(ideal_time, percen25_ir, label='P25', linewidth=1.5, linestyle='dashdot', color='darkorange')
ax_ideal.plot(ideal_time, percen10_ir, label='P10', linewidth=1.5, linestyle='dotted', color='darkorange')

ax_ideal.set_ylabel('$R$ (Reduction factor)')
ax_ideal.set_xlabel('time (Myrs)')
ax_ideal.legend(frameon=False, fontsize=16)
#ax_ideal.set_title("Ideal")

ax_amb.scatter(amb_time, fractions_a, marker='x', color='black', s=8)
ax_amb.plot(amb_time, mean_ar, label='mean', linewidth=1.5, linestyle='-', color='royalblue')
ax_amb.plot(amb_time, median_ar, label='median', linewidth=1.5, linestyle='--', color='darkorange')
ax_amb.plot(amb_time, percen25_ar, label='P25', linewidth=1.5, linestyle='dashdot', color='darkorange')
ax_amb.plot(amb_time, percen10_ar, label='P10', linewidth=1.5, linestyle='dotted', color='darkorange')
ax_amb.set_ylabel('$R$ (Reduction factor)')
ax_amb.set_xlabel('time (Myrs)')
ax_amb.legend(frameon=False, fontsize=16)
#ax_amb.set_title("Ambipolar Diffusion")

fig_ideal.savefig('./time_reduction_ideal.png')
fig_amb.savefig('./time_reduction_amb.png')
plt.close()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig_ideal, ax_ideal = plt.subplots()
fig_amb, ax_amb = plt.subplots()

ax_ideal.scatter(ideal_time, fractions_i, label='fractions', marker='x', color='black')
ax_ideal.plot(ideal_time, mean_ir, label='mean', linewidth=1.5, linestyle='-', color='darkorange')
ax_ideal.plot(ideal_time, median_ir, label='median', linewidth=1.5, linestyle='--', color='darkorange')
ax_ideal.set_ylabel('$R$ (Reduction factor)')
ax_ideal.set_xlabel('time (Myrs)')
ax_ideal.legend(frameon=False)
#ax_ideal.set_title("Ideal")
axins_ideal = inset_axes(ax_ideal, width="20%", height="20%", loc='lower right')
mask_ideal = [t >= 3 for t in ideal_time]
t_zoom = [t for t, m in zip(ideal_time, mask_ideal) if m]
f_zoom = [f for f, m in zip(fractions_i, mask_ideal) if m]
mean_zoom = [m for m, msk in zip(mean_ir, mask_ideal) if msk]
median_zoom = [m for m, msk in zip(median_ir, mask_ideal) if msk]
axins_ideal.scatter(t_zoom, f_zoom, marker='x', color='black')
axins_ideal.plot(t_zoom, mean_zoom, color='darkorange')
axins_ideal.plot(t_zoom, median_zoom, color='darkorange', linestyle='--')
#axins_ideal.set_title("Zoom $t > 3.5$", fontsize=8)
axins_ideal.tick_params(labelsize=8)

ax_amb.scatter(amb_time, fractions_a, label='fractions', marker='x', color='black')
ax_amb.plot(amb_time, mean_ar, label='mean', linewidth=1.5, linestyle='-', color='royalblue')
ax_amb.plot(amb_time, median_ar, label='median', linewidth=1.5, linestyle='--', color='royalblue')
ax_amb.set_ylabel('$R$ (Reduction factor)')
ax_amb.set_xlabel('time (Myrs)')
ax_amb.legend(frameon=False)
#ax_amb.set_title("Ambipolar Diffusion")
axins_amb = inset_axes(ax_amb, width="20%", height="20%", loc='lower right')
mask_amb = [t >= 4.2903 for t in amb_time]
t_zoom = [t for t, m in zip(amb_time, mask_amb) if m]
f_zoom = [f for f, m in zip(fractions_a, mask_amb) if m]
mean_zoom = [m for m, msk in zip(mean_ar, mask_amb) if msk]
median_zoom = [m for m, msk in zip(median_ar, mask_amb) if msk]
axins_amb.scatter(t_zoom, f_zoom, marker='x', color='black')
axins_amb.plot(t_zoom, mean_zoom, color='royalblue')
axins_amb.plot(t_zoom, median_zoom, color='royalblue', linestyle='--')
#axins_amb.set_title("Zoom $t > 3.5$", fontsize=8)
axins_amb.tick_params(labelsize=8)

fig_ideal.savefig('./time_reduction_ideal_win.png')
fig_amb.savefig('./time_reduction_amb_win.png')
plt.close()

import os
os.makedirs('./reduction_density/ideal', exist_ok=True)
os.makedirs('./reduction_density/amb', exist_ok=True)

cur_min = 1.0
mini = 1.0
cur_max =0.0
maxi = 0.0


for i, tup in enumerate(s_ideal):
    r_, x, mean, median, ten, s_size, no, f, n_ = tup
    print(len(r_), len(x), len(mean), len(median), len(ten), len(s_size), no, len(n_))
    r = np.array(r_)
    r = r[r<1]
    t = np.round(ideal_time[i], 6)
    cur_min = f
    if cur_min < mini:
        mini = cur_min
    cur_max = f
    if cur_max > maxi:  
        maxi = cur_max    
    figR, axR = plt.subplots()
    axR.scatter(n_, r_, marker ='x', color='darkorange')
    axR.plot(x, mean, label='mean', linewidth=1.5, linestyle='-', color='darkorange')

    axR.set_xscale('log')
    axR.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'./reduction_density/ideal/ideal_{no}_scatter.png')
    plt.close()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6)) #
    num_bins = len(r)//10
    if num_bins < 10:
        num_bins = 10
    ax0.hist(r, num_bins, density = True)
    ax0.set_xlabel('Reduction factor', fontsize = 20)
    ax0.set_ylabel('PDF', fontsize = 20)
    plt.setp(ax1.get_xticklabels(), fontsize = 16)
    plt.setp(ax1.get_yticklabels(), fontsize = 16)
    ax0.set_title(f'$t$ = {t}  Myrs')
    ax1.plot(x, mean, label='mean', linewidth=1.5, linestyle='-', color='darkorange')
    ax1.plot(x, median, label='median', linewidth=1.5, linestyle='--', color='darkorange')
    ax1.plot(x, ten, label='10th percentile', linewidth=1.5, linestyle='-', color='royalblue')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$R$', fontsize = 16)
    ax1.set_xlabel('$n_g$', fontsize = 16)
    ax1.set_title(f'$f$ = {f}', fontsize = 16)
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'./reduction_density/ideal/ideal_{no}_reduction_density.png')
    plt.close()

print("Ideal: ", mini, maxi)

cur_min = 1.0
mini = 1.0
cur_max =0.0
maxi = 0.0

for i, tup in enumerate(s_amb):
    r_, x, mean, median, ten, s_size, no, f, n_ = tup
    r = np.array(r_)
    r = r[r<1]
    t = np.round(amb_time[i], 6)
    print(len(r_), len(x), len(mean), len(median), len(ten), len(s_size), len(no), len(n_))
    cur_min = f
    if cur_min < mini:
        mini = cur_min
    cur_max = f
    if cur_max > maxi:  
        maxi = cur_max    
    figR, axR = plt.subplots()
    axR.scatter(n_, r_, marker ='x', color='darkorange')
    axR.plot(x, mean, label='mean', linewidth=1.5, linestyle='-', color='darkorange')
    axR.set_xscale('log')
    axR.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'./reduction_density/amb/amb_{no}_scatter.png')
    plt.close()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    num_bins = len(r)//10
    if num_bins < 10:
        num_bins = 10
    t = np.round(amb_time[i], 6)
    ax0.hist(r_, num_bins, density = True)
    ax0.set_xlabel('Reduction factor', fontsize = 20)
    ax0.set_ylabel('PDF', fontsize = 20)
    ax0.set_title(f'$t$ = {t}  Myrs')
    plt.setp(ax1.get_xticklabels(), fontsize = 16)
    plt.setp(ax1.get_yticklabels(), fontsize = 16)
    ax1.plot(x, mean , label='mean', linewidth=1.5, linestyle='-', color='darkorange')
    ax1.plot(x, median, label='median', linewidth=1.5, linestyle='--', color='darkorange')
    ax1.plot(x, ten, label='10th percentile', linewidth=1.5, linestyle='-', color='royalblue')
    ax1.set_xscale('log')
    ax1.set_ylabel('$R$')
    ax1.set_xlabel('$n_g$')
    ax1.set_title(f'$f$ = {f}')
    plt.tight_layout()
    plt.savefig(f'./reduction_density/amb/amb_{no}_reduction_density.png')
    plt.close()

print("Amb: ", mini, maxi)