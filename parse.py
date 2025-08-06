import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict
import numpy as np
import time, glob, re
import csv
import pingouin as pg

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

def pocket_finder(bfield, p_r, B_r, img = '', plot=False):
    """  
    pocket_finder(bfield, p_r, B_r, img=i, plot=False)
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

    if False:
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

def hist3d(data, output):
    """
    data = [
        np.array,
        np.array,
        np.array,
        np.array,
        ...
        ]
    """
    colors = ['yellow', 'blue', 'green', 'red','yellow', 'blue', 'green', 'red']

    # Set up 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each histogram at a different Y position
    bins_ = int(np.ceil(np.sqrt(np.mean([a.shape[0] for a in data]))))
    for i, (d, color) in enumerate(zip(data, colors)):
        counts, bins = np.histogram(d, bins=bins_)
        xs = 0.5 * (bins[:-1] + bins[1:])
        ys = np.full_like(xs, i * 5)  # Stack along Y
        zs = np.zeros_like(xs)

        dx = (bins[1] - bins[0]) * 1.0  # Thinner bars in X
        dy = 0.1                    # Thinner depth in Y
        dz = counts

        ax.bar3d(xs, ys, zs, dx, dy, dz, color=color, alpha=0.6)

    # Labels and aesthetics
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Stacked 3D Histograms')
    ax.view_init(elev=20, azim=-60)  # adjust angle to match reference image
    plt.tight_layout()
    plt.savefig(f'./hist3d_{output}.png')
    plt.close()

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

StatsRones  = OrderedDict({'ideal': [], 'amb': []})
StatsRzero  = OrderedDict({'ideal': [], 'amb': []})

for sim, times in R100_PATH.items():
    for index, time in enumerate(times):
        x_distro     = np.array(X_PATH[sim][time])
        r100_distro  = np.array(R100_PATH[sim][time])
        r10_distro   = np.array(R10_PATH[sim][time])
        n_distro     = np.log10(NR_PATH[sim][time])
        b_distro     = np.array(BS_PATH[sim][time])        

        tulip        = (time, n_distro, r100_distro) 
        #print(time, n_peak, x_distro.shape, r100_distro.shape, r10_distro.shape, n_distro.shape, b_distro.shape)
        ReducedBundle[sim].append(tulip)

        # not ones
        total_at_time = r100_distro.shape[0]
        r = r100_distro[r100_distro<1]
        x = x_distro[r100_distro<1]
        b = b_distro[r100_distro<1]
        n = n_distro[r100_distro<1]
        not_ones_at_time = r.shape[0]
        f_at_time        = not_ones_at_time/total_at_time         

        StatsRzero[sim].append((r,x,b,n, f_at_time))

        # ones
        r = r100_distro[r100_distro==1]
        x = x_distro[r100_distro==1]
        b = b_distro[r100_distro==1]
        n = n_distro[r100_distro==1]

        StatsRones[sim].append((r,x,b,n, 1 - f_at_time ))
        
common_times_ideal = sorted(set(CD_PATH['ideal'].keys()) & set(CD_LOS['ideal'].keys()), key=float)
global_max = np.max([float(t) for t in common_times_ideal])

common_times_amb = sorted(set(CD_PATH['amb'].keys()) & set(CD_LOS['amb'].keys()), key=float)
common_times_amb = [t for t in common_times_amb if float(t) < global_max]


for sim, common_times in zip(['ideal', 'amb'], [common_times_ideal, common_times_amb]):
    for index, time in enumerate(common_times):
        cp_distro     = np.array(CD_PATH[sim][time])
        cl_distro     = np.array(CD_LOS[sim][time])
        x_distro     = np.array(X_PATH[sim][time])
        tulip         = (time, cp_distro, cl_distro)
        #print(f"{index:<5} {time:<20} {sim:<10}")
        ReducedColumn[sim].append(tulip)


if False: # Ideal/Amb Columns PATH & LOS
    common_times, data_path, data_los = zip(*[(float(time), cp_distro, cl_distro) for time, cp_distro, cl_distro in ReducedColumn['ideal']])

    positions_los = np.arange(len(data_los)) # this needs work
    positions_path = positions_los - 0.5

    fig, ax = plt.subplots()
    ax.boxplot(data_los, positions=positions_los, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='skyblue'), label=r'$N_{los}$ shifted')

    ax.boxplot(data_path, positions=positions_path, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='orange'), label=r'$N_{path}$')

    xticks = positions_los 
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(common_times, 4), rotation=70, fontsize=8)

    ax.set_ylabel('Effective Column Density')
    ax.set_xlabel('Time (Myrs)')
    ax.set_title('Column Densities (ideal)')
    ax.set_yscale('log')
    legend_handles = [
        Patch(facecolor='skyblue', label=r'$N_{los}$'),
        Patch(facecolor='orange', label=r'$N_{path}$')
    ]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 1, 1])  # leave space on the right
    plt.savefig("./ideal_l_p.png")
    plt.close()

    common_times, data_path, data_los = zip(*[(float(time), cp_distro, cl_distro) for time, cp_distro, cl_distro in ReducedColumn['amb']])

    positions_los = np.arange(len(data_los))
    positions_path = positions_los - 0.5

    fig, ax = plt.subplots()
    ax.boxplot(data_los, positions=positions_los, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='skyblue'), label=r'$N_{los}$ shifted')

    ax.boxplot(data_path, positions=positions_path, widths=0.2,
            flierprops=dict(marker='|', markersize=2, color='red'),
            patch_artist=True, boxprops=dict(facecolor='orange'), label=r'$N_{path}$')

    xticks = positions_los 
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(common_times, 4), rotation=70, fontsize=8)
    ax.set_xlim(right=29)  # or ax.set_xlim(left=min_time, right=4.2904)
    ax.set_ylabel('Effective Column Density')
    ax.set_xlabel('Time (Myrs)')
    ax.set_title('Column Densities (non-ideal)')
    ax.set_yscale('log')

    legend_handles = [
        Patch(facecolor='skyblue', label=r'$N_{los}$'),
        Patch(facecolor='orange', label=r'$N_{path}$')
    ]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 1, 1])  # leave space on the right
    plt.savefig("./amb_l_p.png")
    plt.close()

gs = 18
func = np.mean

if False: # HexBin Ideal/AMB
    times, r100 = zip(*[(float(time), r100_distro[r100_distro < 1])
                        for time, _, r100_distro in ReducedBundle['ideal']])

    #xy_pairs = [(t, val) for t, vals in zip(times, r100) for val in vals]
    xy_pairs = [(t, val) for t, vals in zip(times, r100) for val in vals]
    x = [pair[0] for pair in xy_pairs]
    y = [pair[1] for pair in xy_pairs]

    xlim = min(x), max(x)
    ylim = 0.0, 1.0
    """
    histogram3d(
        x, y,
        x_bins=30,
        y_bins=30,
        x_range=(min(x), max(x)),
        y_range=(0,1.0),
        xlabel="time (Myrs)",
        ylabel="$R$",
        title="$R$ distribution in time",
        output="ideal"
    )
    """
    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

    hb = ax0.hexbin(x, y, gridsize=gs, cmap='inferno',reduce_C_function=func)#gridsize=50,
    ax0.set(xlim=xlim, ylim=ylim)
    ax0.set_title("Hexagon binning")
    cb = fig.colorbar(hb, ax=ax0, label='counts')

    hb = ax1.hexbin(x, y, gridsize=gs, bins='log', cmap='inferno')
    ax1.set(xlim=xlim, ylim=ylim)
    ax1.set_title("With a log color scale (ideal)")
    cb = fig.colorbar(hb, ax=ax1, label='counts')
    plt.savefig('./ideal_r_t')
    plt.close()

    means = [np.mean(vals) for vals in r100]
    errors = [np.std(vals) for vals in r100]

    plt.errorbar(times, means, yerr=errors, fmt='o', capsize=4)
    plt.xlabel('Time')
    plt.ylabel('Mean ± Std')
    plt.savefig('./ideal_r_t_err')
    plt.close()
    
    times, r100 = zip(*[(float(time), r100_distro[r100_distro < 1])
                        for time, _, r100_distro in ReducedBundle['amb']])

    xy_pairs = [(t, val) for t, vals in zip(times, r100) for val in vals]
    x = [pair[0] for pair in xy_pairs]
    y = [pair[1] for pair in xy_pairs]

    xlim = min(x), max(x)
    ylim = 0.0, 1.0
    time_steps = len(x)
    r_avg_size = np.sqrt(np.mean([len(ys) for ys in r100]))

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

    hb = ax0.hexbin(x, y, gridsize=gs, cmap='inferno',reduce_C_function=func) # gridsize=50
    ax0.set(xlim=xlim, ylim=ylim)
    ax0.set_title("Hexagon binning")
    cb = fig.colorbar(hb, ax=ax0, label='counts')

    hb = ax1.hexbin(x, y, gridsize=gs, bins='log', cmap='inferno')
    ax1.set(xlim=xlim, ylim=ylim)
    ax1.set_title("log color scale (non-ideal)")
    cb = fig.colorbar(hb, ax=ax1, label='counts')
    plt.savefig('./amb_r_t')
    plt.close()
    means = [np.mean(vals) for vals in r100]
    errors = [np.std(vals) for vals in r100]
    plt.errorbar(times, means, yerr=errors, fmt='o', capsize=4)
    plt.xlabel('Time')
    plt.ylabel('Mean ± Std')
    plt.savefig('./amb_r_t_err')
    plt.close()
# _, [min,max], Mean, Variance, Skewness, Kurtosis

#r, x, b, n, f = zip(*[(_r, _x, _b, _n, _f)
#                    for _r, _x, _b, _n, _f in StatsRones['ideal']])

from scipy.stats import skew

if True: # Statistical despcriptors and fraction
    #from scipy.stats import skew, kurtosis


    times, r = zip(*[(float(time), r100_distro)
                            for time, _, r100_distro in ReducedBundle['ideal']])

    r_num, r_bounds, r_means, r_var, r_skew, r_kur = [], [], [], [], [], []
    f = []
    for r_ in r:
        print(type(r_))
        r_ = np.array(r_)
        
        total = r_.shape[0]
        r_ = r_[r_<1]
        nones = r_.shape[0]
        f.append(1-nones/total)
        r_num.append(total)
        r_means.append(np.mean(r_))
        r_var.append(np.var(r_))
        r_skew.append(skew(r_))
        r_kur.append(pg.kurtosis(r_))


    mosaic = [
        ['mean',      'std_dev'],
        ['skewness',  'kurtosis'],
        ['fraction',  'fraction'],  # Span the full row
    ]

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(8, 20), sharex=True)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axs['mean'].plot(times, r_means, marker='o', color=default_colors[0])
    axs['mean'].set_ylabel(r'$\mu$ (Mean)')
    axs['mean'].grid(True)

    axs['std_dev'].plot(times, r_var, marker='s', color=default_colors[1])
    axs['std_dev'].set_ylabel(r'$\sigma$ (Std Dev)')
    axs['std_dev'].grid(True)

    axs['skewness'].plot(times, r_skew, marker='^', color=default_colors[2])
    axs['skewness'].set_ylabel(r'$\gamma$ (Skewness)')
    axs['skewness'].grid(True)

    axs['kurtosis'].plot(times, r_kur, marker='d', color=default_colors[3])
    axs['kurtosis'].set_ylabel(r'$\kappa$ (Kurtosis)')
    axs['kurtosis'].grid(True)

    axs['fraction'].plot(times, f, marker='x', color=default_colors[4])
    axs['fraction'].set_xlabel('Time Step')
    axs['fraction'].set_ylabel(r'$f=\frac{\{R=1\}}{\{R\}}$')
    axs['fraction'].grid(True)

    fig.suptitle('Time Evolution of Statistical Moments ($R<1$)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./ideal_moments.png', dpi=300)
    plt.close()

    times, r = zip(*[(float(time), r100_distro)
                            for time, _, r100_distro in ReducedBundle['amb']])

    r_num, r_bounds, r_means, r_var, r_skew, r_kur = [], [], [], [], [], []
    f = []
    for r_ in r:
        print(type(r_))
        r_ = np.array(r_)
        
        total = r_.shape[0]
        r_ = r_[r_<1]
        nones = r_.shape[0]
        f.append(1-nones/total)
        r_num.append(total)
        r_means.append(np.mean(r_))
        r_var.append(np.var(r_))
        r_skew.append(skew(r_))
        r_kur.append(pg.kurtosis(r_))

    #r, x, b, n, f = zip(*[(_r, _x, _b, _n, _f)
    #                    for _r, _x, _b, _n, _f in StatsRones['amb']])

    #r_flat = np.concatenate(r)
    #r_num, r_bounds, r_means, r_var, r_skew, r_kur = describe(r_flat)

    mosaic = [
        ['mean',      'std_dev'],
        ['skewness',  'kurtosis'],
        ['fraction',  'fraction'],  # Span the full row
    ]

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(8, 20), sharex=True)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axs['mean'].plot(times, r_means, marker='o', color=default_colors[0])
    axs['mean'].set_ylabel(r'$\mu$ (Mean)')
    axs['mean'].grid(True)

    axs['std_dev'].plot(times, r_var, marker='s', color=default_colors[1])
    axs['std_dev'].set_ylabel(r'$\sigma$ (Std Dev)')
    axs['std_dev'].grid(True)

    axs['skewness'].plot(times, r_skew, marker='^', color=default_colors[2])
    axs['skewness'].set_ylabel(r'$\gamma$ (Skewness)')
    axs['skewness'].grid(True)

    axs['kurtosis'].plot(times, r_kur, marker='d', color=default_colors[3])
    axs['kurtosis'].set_ylabel(r'$\kappa$ (Kurtosis)')
    axs['kurtosis'].grid(True)

    axs['fraction'].plot(times, f, marker='x', color=default_colors[4])
    axs['fraction'].set_xlabel('Time Step')
    axs['fraction'].set_ylabel(r'$f=\frac{\{R=1\}}{\{R\}}$')
    axs['fraction'].grid(True)

    fig.suptitle('Time Evolution of Statistical Moments ($R<1$)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./amb_moments.png', dpi=300)
    plt.close()

if True: # 3D histogram of R in time (but time is represented by index)

    times, n, r = zip(*[(float(time), _ , r100_distro)
                            for time, _, r100_distro in ReducedBundle['ideal']])
    data = [np.array(r_[r_<1]) for r_ in r]
    hist3d(data, 'ideal')
        
    r_red, n_ref, mean_vec, median_vec, ten_vec, sample_size, fraction, n_og = statistics_reduction(r, n)



    times, n, r = zip(*[(float(time), _, r100_distro)
                            for time, _, r100_distro in ReducedBundle['amb']])
    data = [np.array(r_[r_<1]) for r_ in r]

    hist3d(data, 'amb')
    r_red, n_ref, mean_vec, median_vec, ten_vec, sample_size, fraction, n_og = statistics_reduction(r, n)


exit()
r, x, b, n, f = zip(*[(_r, _x, _b, _n, _f)
                    for _r, _x, _b, _n, _f in StatsRzero['ideal']])

r_flat = np.concatenate(r)
r_num, r_bounds, r_means, r_var, r_skew, r_kur = describe(r_flat)

plt.figure(figsize=(10, 6))
plt.plot(f, label=r'$\mu$ (Mean)', marker='o')
plt.plot(r_means, label=r'$\mu$ (Mean)', marker='o')
plt.plot(r_var, label=r'$\sigma$ (Std Dev)', marker='s')
plt.plot(r_skew, label=r'$\gamma$ (Skewness)', marker='^')
plt.plot(r_kur, label=r'$\kappa$ (Kurtosis)', marker='d')

plt.xlabel('Time Step')
plt.ylabel('Moment Value')
plt.title('Time Evolution of Statistical Moments ($R<1$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./ideal_moments.png', dpi=300)


gs = 18
func = np.mean
times, r100, n_g = zip(
    *[        (float(time), r100_distro[r100_distro < 1], n_distro[r100_distro < 1])
        for time, n_distro, r100_distro in ReducedBundle["ideal"]    ])

xyz_triads = [
    (t, r, n)
    for t, r_vec, n_vec in zip(times, r100, n_g)
    for r, n in zip(r_vec, n_vec)]

x = [t for t, _, _ in xyz_triads]
y = [r for _, r, _ in xyz_triads]
z = [n for _, _, n in xyz_triads]

times, r100 = zip(*[(float(time), r100_distro)
                    for time, _, r100_distro in ReducedBundle['amb']])

#xy_pairs = [(t, val) for t, vals in zip(times, r100) for val in vals]
xy_pairs = [(t, val) for t, vals in zip(times, r100) for val in vals if val != 1]
x = [pair[0] for pair in xy_pairs]
y = [pair[1] for pair in xy_pairs]

if True:

    xlim = min(x), max(x)
    ylim = 0.0, 1.0

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))

    hb = ax0.hexbin(x, y, gridsize=gs, cmap='inferno',reduce_C_function=func) # gridsize=50
    ax0.set(xlim=xlim, ylim=ylim)
    ax0.set_title("Hexagon binning")
    cb = fig.colorbar(hb, ax=ax0, label='counts')

    hb = ax1.hexbin(x, y, gridsize=gs, bins='log', cmap='inferno')
    ax1.set(xlim=xlim, ylim=ylim)
    ax1.set_title("log color scale (non-ideal)")
    cb = fig.colorbar(hb, ax=ax1, label='counts')
    plt.savefig('./amb_r_t_n')


# Generate data
#x = np.linspace(0, 10, 10000)              # Uniformly spaced x-values (e.g. time)
#y = np.random.uniform(0, 10, 10000)        # Random y-values

# Create the hexbin plot
plt.figure(figsize=(10, 8))
plt.hexbin(x, y, gridsize=50, mincnt=1)    # gridsize controls hexagon resolution
plt.colorbar(label='Count in bin')
plt.xlabel('Time (t)')
plt.ylabel('Value (val)')
plt.title('Hexbin Plot with Increased Sample Size')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('./hex_bin_ex.png')