import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict
import numpy as np
import sys, time, glob, re, os
import warnings, csv
import random

"""
This code will take 200 GB of data and summarize it for further analysis

Ideal and Non-Ideal

(t, snap, Rs, Ns, Bs, N_path, ...?)


"""
def dummy_data():
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
    R10      = []
    R100     = []
    Numb100  = []
    B100     = []
    RBundle  = []
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
            B100.append(B_r)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)
            else:
                # NO!
                R = 1.
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)
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
        #RBundle.append((R10, R100, Numb100, B100))

    return R10, R100, Numb100, B100

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

pc_to_cm = 3.086 * 10e+18  # cm per parsec

amb_bundle1   = sorted(glob.glob('./thesis_stats/amb/*/DataBundle*.npz'))
ideal_bundle1 = sorted(glob.glob('./thesis_stats/ideal/*/DataBundle*.npz'))
bundle_dirs = [ideal_bundle1,amb_bundle1]

Bundle = OrderedDict({'ideal': [], 'amb': []})

R100_PATH = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
R10_PATH  = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
NR_PATH   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
CD_PATH   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
BS_PATH   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
X_PATH    = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
CD_LOS    = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
X_LOS     = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})

peak_den_los    = OrderedDict({'ideal': [], 'amb': []})
snap_values_los = OrderedDict({'ideal': [], 'amb': []})
time_values_los = OrderedDict({'ideal': [], 'amb': []})

peak_den_path    = OrderedDict({'ideal': [], 'amb': []})
snap_values_path = OrderedDict({'ideal': [], 'amb': []})
time_values_path = OrderedDict({'ideal': [], 'amb': []})

common_times    = OrderedDict({'ideal': [], 'amb': []})

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
                    repeated.add(snap)
                    snap_values_path[case].append(str(row[0]))
                    time_values_path[case].append(float(row[1]))
                    peak_den_path[case].append(float(row[-1]))
                    continue

        data = np.load(snap_data, mmap_mode='r')
        column_density = data['column_density']
        starting_position = data['starting_point']
        numb_densities = data['number_densities']
        magnetic_fields = data['magnetic_fields']
        threshold = data['thresholds']

        r_10, r_100, n_r, B_r = evaluate_reduction(magnetic_fields, numb_densities, threshold)
        
        snap_columns_sliced = []
        for i in range(column_density.shape[1]):
            snap_columns_sliced += [np.max(column_density[:, i])]
        t = str(time_values_path[case][-1])
        X_r = list(starting_position)
        X_PATH[case][t]    = X_PATH[case].get(t,  list(X_r*0)) + X_r
        BS_PATH[case][t]   = BS_PATH[case].get(t,  list(B_r*0)) + B_r
        CD_PATH[case][t]   = CD_PATH[case].get(t, list(snap_columns_sliced*0)) + snap_columns_sliced
        R10_PATH[case][t]  = R10_PATH[case].get(t,  list(r_10*0)) + list(r_10)
        R100_PATH[case][t] = R100_PATH[case].get(t, list(r_100*0)) + list(r_100)
        NR_PATH[case][t]   = NR_PATH[case].get(t, list(n_r*0))+ list(n_r)

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

amb_bundle2   = sorted(glob.glob('./thesis_los/N/amb/*/DataBundle*.npz'))
ideal_bundle2 = sorted(glob.glob('./thesis_los/N/ideal/*/DataBundle*.npz'))
bundle_dirs = [ideal_bundle2, amb_bundle2]

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
                    repeated.add(snap)
                    snap_values_los[case].append(str(row[0]))
                    time_values_los[case].append(float(row[1]))
                    peak_den_los[case].append(float(row[-1]))
                    if time_values_los[case][-1] in time_values_path[case]:
                        common_times[case].append(time_values_los[case][-1])
                    continue

        data = np.load(snap_data, mmap_mode='r') 
        column_density = data['column_densities']*pc_to_cm
        numb_densities = data['number_densities']

        snap_columns_sliced = []
        for i in range(column_density.shape[1]):
            snap_columns_sliced += [np.max(column_density[:, i])]
        
        t = str(time_values_los[case][-1])
        CD_LOS[case][t] = CD_LOS[case].get(str(row[1]), snap_columns_sliced * 0) + snap_columns_sliced

"""
Data obtaines up to this points is:

CD_PATH
R100_PATH
R10_PATH
NR_PATH
snap_values_path
time_values_path
peak_den_path

CD_LOS
snap_values_los
time_values_los
peak_den_los


for both ideal and non-ideal MHD
"""

ReducedBundle = OrderedDict({'ideal': [], 'amb': []})
ReducedColumn = OrderedDict({'ideal': [], 'amb': []})

for sim, times in R100_PATH.items():
    for index, time in enumerate(times):
        n_peak       = np.log10(peak_den_path[sim][index])
        x_distro     = np.array(X_PATH[sim][time])
        r100_distro  = np.array(R100_PATH[sim][time])
        r10_distro   = np.array(R10_PATH[sim][time])
        b_distro     = np.array(BS_PATH[sim][time])
        n_distro     = np.log10(NR_PATH[sim][time])
        tulip        = (time, n_peak, r100_distro, x_distro, r10_distro, n_distro, b_distro) 
        #print(time, n_peak, x_distro.shape, r100_distro.shape, r10_distro.shape, n_distro.shape, b_distro.shape)
        ReducedBundle[sim].append(tulip)
        
common_times_ideal = sorted(set(CD_PATH['ideal'].keys()) & set(CD_LOS['ideal'].keys()), key=float)
common_times_amb   = sorted(set(CD_PATH['amb'].keys()) & set(CD_LOS['amb'].keys()), key=float)

for sim, common_times in zip(['ideal', 'amb'], [common_times_ideal, common_times_amb]):
    for index, time in enumerate(common_times):
        cp_distro     = np.array(CD_PATH[sim][time])
        cl_distro     = np.array(CD_LOS[sim][time])
        x_distro     = np.array(X_PATH[sim][time])
        tulip         = (time, cp_distro, cl_distro, x_distro)
        #print(f"{index:<5} {time:<20} {sim:<10}")
        ReducedColumn[sim].append(tulip)


if True: 
    common_times, data_path, data_los = zip(*[(float(time), cp_distro, cl_distro) for time, cp_distro, cl_distro in ReducedColumn['ideal']])
    positions_los = common_times_ideal#np.arange(len(data_los))
    positions_path = positions_los*(0.95)#positions_los - 0.25


    fig, ax = plt.subplots()
    ax.boxplot(data_los, positions=positions_los, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='skyblue'), label=r'$N_{los}$ shifted')

    ax.boxplot(data_path, positions=positions_path, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='orange'), label=r'$N_{path}$')

    xticks = positions_los 
    ax.set_xticks(xticks)
    ax.set_xticklabels(common_times, rotation=60)

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
    plt.tight_layout(rect=[0, 0, 1, 1])  # leave space on the right
    plt.savefig("./ideal_l_p.png")
    plt.close()

if True: 
    common_times, data_path, data_los = zip(*[(float(time), cp_distro, cl_distro) for time, cp_distro, cl_distro in ReducedColumn['amb']])
    
    positions_los = common_times_amb#np.arange(len(data_los))
    positions_path = positions_los*(0.95)#positions_los - 0.25

    fig, ax = plt.subplots()
    ax.boxplot(data_los, positions=positions_los, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='skyblue'), label=r'$N_{los}$ shifted')

    ax.boxplot(data_path, positions=positions_path, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='orange'), label=r'$N_{path}$')

    #xticks = positions_los 
    #ax.set_xticks(xticks)
    #ax.set_xticklabels(common_times, rotation=60)

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
    plt.savefig("./amb_l_p.png")
    plt.close()

exit()
times, r100 = zip(*[(float(time), r) for time, r, _ in ReducedColumn['ideal']])

if True:

    x = times # time
    y = r100# reduction factor
    xlim = x.min(), x.max()
    ylim = y.min(), y.max()

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

    hb = ax0.hexbin(x, y, gridsize=50, cmap='inferno')
    ax0.set(xlim=xlim, ylim=ylim)
    ax0.set_title("Hexagon binning")
    cb = fig.colorbar(hb, ax=ax0, label='counts')

    hb = ax1.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
    ax1.set(xlim=xlim, ylim=ylim)
    ax1.set_title("With a log color scale (ideal)")
    cb = fig.colorbar(hb, ax=ax1, label='counts')
    plt.savefig('./ideal_r_t')

times, r100 = zip(*[(float(time), r) for time, r, _ in ReducedColumn['amb']])

if True:

    x = times # time
    y = r100# reduction factor
    xlim = x.min(), x.max()
    ylim = y.min(), y.max()

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

    hb = ax0.hexbin(x, y, gridsize=50, cmap='inferno')
    ax0.set(xlim=xlim, ylim=ylim)
    ax0.set_title("Hexagon binning")
    cb = fig.colorbar(hb, ax=ax0, label='counts')

    hb = ax1.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')
    ax1.set(xlim=xlim, ylim=ylim)
    ax1.set_title("With a log color scale (non-ideal)")
    cb = fig.colorbar(hb, ax=ax1, label='counts')
    plt.savefig('./ideal_r_t')
