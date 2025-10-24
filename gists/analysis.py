import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict
import numpy as np
import sys, time, glob, re, os
import warnings, csv
import random


#...............Core Density Comparison..................


cases = ['ideal', 'amb']

s_ideal = []
s_amb = []
t_ideal = []
t_amb = []
nc_ideal = []
nc_amb = []

for case in cases:
    file_path = f'./{case}_cloud_trajectory.txt'
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if case == "ideal":
                s_ideal.append(str(row[0]))
                t_ideal.append(float(row[1]))
                nc_ideal.append(float(row[-1]))

            if case == "amb":
                s_amb.append(str(row[0]))
                t_amb.append(float(row[1]))
                nc_amb.append(float(row[-1]))

mu_mH = 2.35 * 1.67e-24
rho_ideal = np.array(nc_ideal) * mu_mH
rho_amb = np.array(nc_amb) * mu_mH
fig, ax1 = plt.subplots()
ax1.set_xlabel(r'$t$ (Myrs)')
ax1.set_ylabel(r'$n_g$ (cm$^{-3}$)', fontsize=12)
ax1.set_yscale('log')
ax1.plot(t_ideal, nc_ideal, label='Core Density (ideal MHD)', color='dodgerblue')
ax1.plot(t_amb, nc_amb, label='Core Density (non-ideal MHD)', color='darkorange')
ax1.tick_params(axis='y')
ax1.legend(loc='center')
plt.grid()
ax2 = ax1.twinx()
ax2.set_ylabel(r'$\rho$ (g cm$^{-3}$)', fontsize=12)
ax2.set_yscale('log')
ax2.set_ylim(ax1.get_ylim()[0] * mu_mH, ax1.get_ylim()[1] * mu_mH)
plt.title('Core Density Evolution')
fig.tight_layout()
plt.savefig('./images/core_den.png')
plt.close()

#...............Reduction Factor Statistics..................

pc_to_cm = 3.086 * 10e+18  # cm per parsec

start_time = time.time()

amb_bundle   = sorted(glob.glob('./thesis_stats/amb/*/DataBundle*.npz'))
ideal_bundle = sorted(glob.glob('./thesis_stats/ideal/*/DataBundle*.npz'))

bundle_dirs = [ideal_bundle,amb_bundle]

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
    R = np.array(R)
    N = np.array(N)
    # R is numpy array
    def _stats(n, d_data, r_data, p_data=0):
        sample_r = []

        for i in range(0, len(d_data)):
            if np.abs(np.log10(d_data[i]/n)) < 1/8:
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

    x_n = np.logspace(minimum, maximum, Npoints)
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

R100_PATH = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
R10_PATH  = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
NR_PATH   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
CD_PATH   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
Delta     = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
delta = 0.0

peak_den_path    = OrderedDict({'ideal': [], 'amb': []})
snap_values_path = OrderedDict({'ideal': [], 'amb': []})
time_values_path = OrderedDict({'ideal': [], 'amb': []})

for bundle_dir in bundle_dirs: # ideal and ambipolar
    if bundle_dir == []:
        continue
    case = str(bundle_dir[0].split('/')[-3])
    repeated = set()
    for snap_data in bundle_dir: # from 000 to 490 
        snap = str(snap_data.split('/')[-2])

        file_path = f'./{case}_cloud_trajectory.txt'

        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if snap == str(row[0]) and snap not in repeated:
                    if float(row[1]) > 4.3:
                        break
                    repeated.add(snap)
                    snap_values_path[case].append(str(row[0]))
                    time_values_path[case].append(float(row[1]))
                    peak_den_path[case].append(float(row[-1]))
                    continue

        data = np.load(snap_data, mmap_mode='r')
        column_density = data['column_density']
        radius_vector = data['positions']
        trajectory = data['trajectory']
        numb_densities = data['number_densities']
        magnetic_fields = data['magnetic_fields']
        threshold = data['thresholds']

        r_bundle, r_10, r_100, n_r = evaluate_reduction(magnetic_fields, numb_densities, threshold)
        snap_columns_sliced = []

        for i in range(column_density.shape[1]):
            snap_columns_sliced += [np.max(column_density[:, i])]
        t = str(time_values_path[case][-1])
        CD_PATH[case][t] = CD_PATH[case].get(t, snap_columns_sliced * 0) + snap_columns_sliced
        R10_PATH[case][t]  =  R10_PATH[case].get(t,  list(r_10*0)) + list(r_10)
        R100_PATH[case][t] = R100_PATH[case].get(t, list(r_100*0)) + list(r_100)
        NR_PATH[case][t] = NR_PATH[case].get(t, list(n_r*0))+ list(n_r)

"""
Data obtaines up to this points is:

CD_PATH
R100_PATH
R10_PATH
NR_PATH
snap_values_path
time_values_path
peak_den_path

for both ideal and non-ideal MHD
"""

amb_bundle   = sorted(glob.glob('./thesis_los/N/amb/*/DataBundle*.npz'))
ideal_bundle = sorted(glob.glob('./thesis_los/N/ideal/*/DataBundle*.npz'))

bundle_dirs = [ideal_bundle, amb_bundle]

peak_den_los    = OrderedDict({'ideal': [], 'amb': []})
snap_values_los = OrderedDict({'ideal': [], 'amb': []})
time_values_los = OrderedDict({'ideal': [], 'amb': []})
CD_LOS = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})

for bundle_dir in bundle_dirs:  # ideal and ambipolar

    repeated = set()
    if bundle_dir == []:
        continue
    case = str(bundle_dir[0].split('/')[-3])
    
    for snap_data in bundle_dir:  # from 000 to 490 
        snap = str(snap_data.split('/')[-2])
        if int(snap) > 400:
            continue
        file_LOS = f'./{case}_cloud_trajectory.txt'
        with open(file_LOS, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if snap == str(row[0]) and snap not in repeated:
                    if case == 'ideal':
                        repeated.add(snap)
                        snap_values_los[case].append(str(row[0]))
                        time_values_los[case].append(float(row[1]))
                        peak_den_los[case].append(float(row[-1]))
                        continue

                    if case == 'amb':
                        repeated.add(snap)
                        snap_values_los[case].append(str(row[0]))
                        time_values_los[case].append(float(row[1]))
                        peak_den_los[case].append(float(row[-1]))

                        continue

        data = np.load(snap_data, mmap_mode='r')
        threshold = data['thresholds']
        threshold_rev = data['thresholds_rev']        
        column_density = data['column_densities']*pc_to_cm
        radius_vector = data['positions']
        numb_densities = data['number_densities']

        snap_columns_sliced = []
        for i in range(column_density.shape[1]):
            snap_columns_sliced += [np.max(column_density[:, i])]
        
        CD_LOS[case][str(row[1])] = CD_LOS[case].get(str(row[1]), snap_columns_sliced * 0) + snap_columns_sliced

if True:

    time_size = 30
    times = [f"{0.1 * i:.1f}" for i in range(time_size)]

    data_size = 2000    

    def make_uniform_dict():
        return OrderedDict({
            'ideal': OrderedDict({t: [1-np.random.beta(a=2, b=5)  for _ in range(data_size)] for t in times}),
            'amb':   OrderedDict({t: [1-np.random.beta(a=2, b=5)  for _ in range(data_size)] for t in times})
        })

    def make_logspace_dict(low_exp, high_exp):
        return OrderedDict({
            'ideal': OrderedDict({
                t: list(np.logspace(low_exp, high_exp, num=data_size))
                for t in times
            }),
            'amb': OrderedDict({
                t: list(np.logspace(low_exp, high_exp, num=data_size))
                for t in times
            })
        })

    R100_PATH = make_uniform_dict()
    R10_PATH  = make_uniform_dict()

    NR_PATH = make_logspace_dict(2, 7)    # 10^2 to 10^7
    CD_PATH = make_logspace_dict(19, 23)  # 10^19 to 10^27
    CD_LOS = make_logspace_dict(23, 27)  # 10^19 to 10^27

"""
Data obtaines up to this points is:

CD_LOS
snap_values_los
time_values_los
peak_den_los

for both ideal and non-ideal MHD
"""

# in the test I have nothing for CD_LOS so it'll be empty
common_ideal_keys = CD_LOS['ideal'].keys() & CD_PATH['ideal'].keys() # times ideal in common 
common_amb_keys   = CD_LOS['amb'].keys() & CD_PATH['amb'].keys()     # times amb   in common 

ideal_time_cd = [np.round(float(k),6) for k in common_ideal_keys]
amb_time_cd   = [np.round(float(k),6) for k in common_amb_keys] 

#CD_LOS['ideal'] = OrderedDict((k, CD_LOS['ideal'][k]) for k in common_ideal_keys)
#CD_PATH['ideal'] = OrderedDict((k, CD_PATH['ideal'][k]) for k in common_ideal_keys)

#CD_LOS['amb'] = OrderedDict((k, CD_LOS['amb'][k]) for k in common_amb_keys)
#CD_PATH['amb'] = OrderedDict((k, CD_PATH['amb'][k]) for k in common_amb_keys)



data_los = list(CD_LOS['amb'].values())  
data_path = list(CD_PATH['amb'].values())

positions_los = np.arange(len(data_los))
positions_path = positions_los - 0.25

fig, ax = plt.subplots()
ax.boxplot(data_los, positions=positions_los, widths=0.2,
           flierprops=dict(marker='|', markersize=2, color='red'),
           patch_artist=True, boxprops=dict(facecolor='skyblue'), label=r'$N_{los}$ shifted')

ax.boxplot(data_path, positions=positions_path, widths=0.2,
           flierprops=dict(marker='|', markersize=2, color='red'),
           patch_artist=True, boxprops=dict(facecolor='orange'), label=r'$N_{path}$')

xticks = positions_los 
ax.set_xticks(xticks)
ax.set_xticklabels(amb_time_cd, rotation=60)

ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title('Column Densities (non-ideal)')
ax.set_yscale('log')
ax.grid(True)
legend_handles = [
    Patch(facecolor='skyblue', label=r'$N_{los}$'),
    Patch(facecolor='orange', label=r'$N_{path}$')
]
ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(rect=[0, 0, 1, 1])  # leave space on the right
plt.savefig("./images/amb_los_path.png")
plt.close()



data_path = list(CD_PATH['ideal'].values())  
data_los = list(CD_LOS['ideal'].values())  

positions_path = np.arange(len(data_path)) 
positions_los = positions_path - 0.25

fig, ax = plt.subplots()
ax.boxplot(data_los, positions=positions_los, widths=0.2,
           flierprops=dict(marker='|', markersize=2, color='red'),
           patch_artist=True, boxprops=dict(facecolor='skyblue'), label=r'$N_{los}$ shifted')

ax.boxplot(data_path, positions=positions_path, widths=0.2,
           flierprops=dict(marker='|', markersize=2, color='red'),
           patch_artist=True, boxprops=dict(facecolor='orange'), label=r'$N_{path}$')

xticks = positions_los 
ax.set_xticks(xticks)
ax.set_xticklabels(ideal_time_cd, rotation=60)

ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title('Column Densities (ideal)')
ax.set_yscale('log')
ax.grid(True)
legend_handles = [
    Patch(facecolor='skyblue', label=r'$N_{los}$'),
    Patch(facecolor='orange', label=r'$N_{path}$')
]
ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(rect=[0, 0, 1, 1]) 
plt.savefig("./images/ideal_los_path.png")
plt.close()


params = list(R100_PATH['amb'].values())
densities = list(NR_PATH['amb'].values())

T_flat = []
R_flat = []
D_flat = []

for t_idx in range(len(times)):
    R_list = params[t_idx]
    D_list = densities[t_idx]
    
    T_flat.extend([times[t_idx]] * len(R_list))
    R_flat.extend(R_list)
    D_flat.extend(D_list)

T_flat = np.array(T_flat)
R_flat = np.array(R_flat)
D_flat = np.array(D_flat)

plt.figure(figsize=(10, 5))
sc = plt.scatter(T_flat, R_flat, c=D_flat, cmap='viridis', norm=LogNorm(), s=20, marker='s')


plt.colorbar(sc, label='Density (cm$^{-3}$)')
plt.xlabel('Time (s)')
plt.ylabel('R (adimensional)')
plt.title('Scatter Plot of R vs Time Colored by Density')
plt.tight_layout()
plt.savefig('./images/scatter_t_r_d_amb.png')
plt.close()

time = [float(t) for t in R100_PATH['ideal'].keys()]
params = list(R100_PATH['ideal'].values())
densities = list(NR_PATH['ideal'].values())


T_flat = []
R_flat = []
D_flat = []

for t_idx in range(len(time)):
    R_list = params[t_idx]
    D_list = densities[t_idx]
    
    T_flat.extend([time[t_idx]] * len(R_list))
    R_flat.extend(R_list)
    D_flat.extend(D_list)

T_flat = np.array(T_flat)
R_flat = np.array(R_flat)
D_flat = np.array(D_flat)

plt.figure(figsize=(10, 5))

sc = plt.scatter(T_flat, R_flat, c=D_flat, cmap='viridis', norm=LogNorm(), s=20, marker='s')

plt.colorbar(sc, label='Density (cm$^{-3}$)')
plt.xlabel('Time (s)')
plt.ylabel('R (adimensional)')
plt.title('Scatter Plot of R vs Time Colored by Density')
plt.tight_layout()
plt.savefig('./images/scatter_t_r_d_ideal.png')
plt.close()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import numpy as np

fig, ax = plt.subplots()

# Normalize the colors
norm = LogNorm(vmin=np.min(D_flat), vmax=np.max(D_flat))
cmap = plt.cm.viridis


for x, y, d in zip(T_flat, R_flat, D_flat):
    color = cmap(norm(d))
    rect = patches.Rectangle((x, y), 1/10, 1/4000, facecolor=color, edgecolor='none')
    ax.add_patch(rect)

# Set limits to contain all rectangles
ax.set_xlim(min(T_flat) - 0.1, max(T_flat) + 0.2)
ax.set_ylim(min(R_flat) - 0.05, max(R_flat) + 0.05)

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='D')

plt.xlabel('T')
plt.ylabel('R')
plt.savefig('./images/__.png')
plt.close()


time = [float(t) for t in R10_PATH['ideal'].keys()]
params = list(R10_PATH['ideal'].values())

T_flat = []
R_flat = []

for t_idx, R_list in enumerate(params):
    T_flat.extend([time[t_idx]] * len(R_list))
    R_flat.extend(R_list)

T_flat = np.array(T_flat)
R_flat = np.array(R_flat)

plt.figure(figsize=(10, 5))
plt.hist2d(
    T_flat, R_flat,
    bins=[len(time), 10],
    cmap='viridis'
)

plt.colorbar(label='Count of R values')
plt.xlabel('Time (s)')
plt.ylabel('R (adimensional)')
plt.title('Heatmap of R Distribution Over Time')
plt.tight_layout()
plt.savefig('./images/hmap_t_r_ideal.png')
plt.close()


time = [float(t) for t in R10_PATH['amb'].keys()]
params = list(R10_PATH['amb'].values())

T_flat = []
R_flat = []

for t_idx, R_list in enumerate(params):
    T_flat.extend([time[t_idx]] * len(R_list))
    R_flat.extend(R_list)

T_flat = np.array(T_flat)
R_flat = np.array(R_flat)

plt.figure(figsize=(10, 5))
plt.hist2d(
    T_flat, R_flat,
    bins=[len(time), 10],
    cmap='viridis'
)

plt.colorbar(label='Count of R values')
plt.xlabel('Time (s)')
plt.ylabel('R (adimensional)')
plt.title('Heatmap of R Distribution Over Time')
plt.tight_layout()
plt.savefig('./images/hmap_t_r_amb.png')
plt.close()



mean_params = []
medi_params = []
dens_distro = []
for t, t_key in enumerate(time):

    r, x_n, mean_vec, median_vec, ten_vec, sample_size, f, _ = statistics_reduction(params[t], densities[t])
    mean_params.append(mean_vec)
    medi_params.append(median_vec)
    dens_distro.append(x_n)

print(np.array(dens_distro).shape)  # Should be (num_times, num_R)
print(np.array(params[0]).shape)    # Should be (num_R,)
print(np.array(params[1]).shape)    # Should be (num_R,)
print(np.array(time).shape)         # Should be (num_times,)

num_times = len(time)                  # Number of time steps
num_params = len(params)               # Number of R values  

# Create the heatmap
plt.figure(figsize=(10, 5))
img = plt.hist2d(time, params, aspect='auto', origin='lower', 
                 extent=[time[0], time[-1], params[0], params[-1]],
                 cmap='viridis', norm=plt.matplotlib.colors.LogNorm())

plt.colorbar(label='Density (cm$^{-3}$)')
plt.xlabel('Time (s)')
plt.ylabel('R (adimensional)')
plt.title('Heatmap of Density Over Time and R')
plt.tight_layout()
plt.savefig('./images/hmap_t_r_f.png')
plt.close()

# Simulate 30 distributions with varying sample sizes and means
np.random.seed(42)
n_distributions = 30
sample_sizes = np.random.randint(100, 1000, size=n_distributions)
means = np.linspace(-3, 3, n_distributions)
stds = np.linspace(0.5, 1.5, n_distributions)

distributions = [np.random.normal(loc=mu, scale=std, size=n)
                 for mu, std, n in zip(means, stds, sample_sizes)]

# Set common bin range for PDF heatmap
x_min, x_max = -6, 6
n_bins = 100
x_bins = np.linspace(x_min, x_max, n_bins)

# Calculate normalized histograms (PDFs)
pdfs = []
for dist in distributions:
    hist, _ = np.histogram(dist, bins=x_bins, density=True)
    pdfs.append(hist)
pdfs = np.array(pdfs)  # Shape: (30, 99)

# Prepare data for 2D histogram and scatter plot
times = []
values = []
for i, dist in enumerate(distributions):
    times.extend([i] * len(dist))
    values.extend(dist)
times = np.array(times)
values = np.array(values)

# Create plots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# 1. PDF Heatmap
im = axs[0].imshow(pdfs, aspect='auto', origin='lower',
                   extent=[x_min, x_max, 0, n_distributions],
                   cmap='viridis')
axs[0].set_title('Heatmap of PDFs Over Time')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Time Index')
fig.colorbar(im, ax=axs[0], label='PDF Value')

# 2. 2D Histogram
h = axs[1].hist2d(times, values, bins=[n_distributions, n_bins],
                  range=[[0, n_distributions], [x_min, x_max]],
                  cmap='plasma', density=True)
axs[1].set_title('2D Histogram of Sample Density Over Time')
axs[1].set_xlabel('Time Index')
axs[1].set_ylabel('X')
fig.colorbar(h[3], ax=axs[1], label='Density')

# 3. Scatter Plot
axs[2].scatter(times, values, alpha=0.2, s=5)
axs[2].set_title('Scatter Plot of Raw Samples Over Time')
axs[2].set_xlabel('Time Index')
axs[2].set_ylabel('X')

plt.tight_layout()
plt.savefig('./images/smth.png', dpi=300)
plt.close()

n_bins = 100  # Number of bins along frequency axis

# Extract and sort time keys
time_keys = sorted(R100_PATH[case].keys(), key=lambda x: float(x))
time_values = [float(t) for t in time_keys]

# Prepare frequency bins
freq_bins = np.linspace(0, 1, n_bins + 1)
freq_centers = 0.5 * (freq_bins[:-1] + freq_bins[1:])

# Initialize parameter_values 2D array
parameter_values = np.zeros((n_bins, len(time_keys)))

# Fill heatmap matrix
for i, t_str in enumerate(time_keys):
    R_vals = np.array(R100_PATH[case][t_str])
    densities = np.array(NR_PATH[case][t_str])

    # Bin the R values and average densities in each bin
    bin_indices = np.digitize(R_vals, freq_bins) - 1  # digitize gives indices from 1
    for j in range(n_bins):
        in_bin = densities[bin_indices == j]
        parameter_values[j, i] = in_bin.mean() if len(in_bin) > 0 else 0.0

# Plot heatmap
plt.figure(figsize=(12, 5))
plt.imshow(parameter_values, aspect='auto', origin='lower',
           extent=[time_values[0], time_values[-1], freq_bins[0], freq_bins[-2]],
           cmap='viridis')

plt.title("Heatmap of Parameter Over Time and Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (R distribution)")
cbar = plt.colorbar()
cbar.set_label("Density")

plt.tight_layout()
plt.savefig('./hmap.png')
plt.close()



# Simulated data
time = np.linspace(0, 10, 500)         # 500 time points
frequency = np.linspace(0, 500, 250)   # 250 frequency points
densities = np.random.rand(len(frequency), len(time))  # Random heatmap data

# Create the plot
plt.figure(figsize=(12, 5))
plt.imshow(parameter_values, aspect='auto', origin='lower',
           extent=[time[0], time[-1], frequency[0], frequency[-1]],
           cmap='viridis')

# Add labels and title
plt.title("Heatmap of Parameter Over Time and Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("Parameter Value")

plt.tight_layout()
plt.savefig('./heatmap_t_r_n.png')
plt.close()

mean_ideal_path   = []
median_ideal_path = []
mean_amb_path     = []
median_amb_path   = []

ideal_snap_path  = []
amb_snap    = []

s_ideal_path = []
s_amb_path   = []

fractions_i = []
fractions_a = []



for k, v in sorted(R10['ideal'].items()):
    mean_ideal.append(np.mean((np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])))
    median_ideal.append(np.median((np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])))

    ideal_snap.append(int(k))
    r_, x, mean, median, ten, s_size, f, n_ = statistics_reduction(np.array(R10['ideal'][k]), np.array(NR['ideal'][k])) 
    s_ideal.append((r_, x, mean, median, ten, s_size, k, f, n_))
    fractions_i.append(float(f))

for k, v in sorted(R10['amb'].items()):
    mean_amb.append(np.mean((np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])))
    median_amb.append(np.median((np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])))

    amb_snap.append(int(k))
    r_, x, mean, median, ten, s_size, f, n_ = statistics_reduction(np.array(R10['amb'][k]), np.array(NR['amb'][k])) 
    s_amb.append((r_, x, mean, median, ten, s_size, k, f, n_))
    fractions_a.append(float(f))



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
ax_ideal.legend(frameon=False, fontsize=10)
#ax_ideal.set_title("Ideal")

ax_amb.scatter(amb_time, fractions_a, marker='x', color='black', s=8)
ax_amb.plot(amb_time, mean_ar, label='mean', linewidth=1.5, linestyle='-', color='royalblue')
ax_amb.plot(amb_time, median_ar, label='median', linewidth=1.5, linestyle='--', color='darkorange')
ax_amb.plot(amb_time, percen25_ar, label='P25', linewidth=1.5, linestyle='dashdot', color='darkorange')
ax_amb.plot(amb_time, percen10_ar, label='P10', linewidth=1.5, linestyle='dotted', color='darkorange')
ax_amb.set_ylabel('$R$ (Reduction factor)')
ax_amb.set_xlabel('time (Myrs)')
ax_amb.legend(frameon=False, fontsize=10)
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
    #print(len(r_), len(x), len(mean), len(median), len(ten), len(s_size), no, len(n_))
    r = np.array(r_)
    r = r[r<1]
    t = np.round(ideal_time[i], 6)
    f = np.round(f, 6)
    #    if t > 4.:
    #        continue 
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
    ax1.scatter(n_, r_, marker ='x', color='dimgrey',alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$R$', fontsize = 16)
    ax1.set_xlabel('$n_g$', fontsize = 16)
    ax1.set_title(f'$f$ = {f}', fontsize = 16)
    ax1.legend()
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
    f = np.round(f, 6)

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
    f = np.round(f, 6)
    ax0.hist(r_, num_bins, density = True)
    
    ax0.set_xlabel('Reduction factor', fontsize = 20)
    ax0.set_ylabel('PDF', fontsize = 20)
    ax0.set_title(f'$t$ = {t}  Myrs')
    plt.setp(ax1.get_xticklabels(), fontsize = 20)
    plt.setp(ax1.get_yticklabels(), fontsize = 20)
    ax1.plot(x, mean , label='mean', linewidth=1.5, linestyle='-', color='darkorange')
    ax1.plot(x, median, label='median', linewidth=1.5, linestyle='--', color='darkorange')
    ax1.plot(x, ten, label='10th percentile', linewidth=1.5, linestyle='-', color='royalblue')
    ax1.scatter(n_, r_, marker ='x', color='dimgrey', alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_ylabel('$R$')
    ax1.set_xlabel('$n_g$', fontsize = 20)
    ax1.set_title(f'$f$ = {f}')
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'./reduction_density/amb/amb_{no}_reduction_density.png')
    plt.close()

print("Amb: ", mini, maxi)

