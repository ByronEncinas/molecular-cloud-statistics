from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import sys, time, glob, re
import warnings, csv


start_time = time.time()

amb_bundle   = sorted(glob.glob('./thesis_stats/amb/*/DataBundle*.npz'))
ideal_bundle = sorted(glob.glob('./thesis_stats/ideal/*/DataBundle*.npz'))

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
            if np.abs(np.log10(d_data[i]/n)) < 1:
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
    ncrit = 100
    print(len(N), len(R))
    mask = R<1
    R = R[mask]
    N = N[mask]
    f = ones/total
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

    rdcut = []
    for i in range(0, Npoints):

        if N[i] > ncrit: 
            rdcut = rdcut + [R[i]]

    return rdcut, x_n, mean_vec, median_vec, ten_vec, sample_size, f

R100 = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
R10  = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
NR   = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
Delta = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})
delta = 0.0

peak_den    = OrderedDict({'ideal': [], 'amb': []})
snap_values = OrderedDict({'ideal': [], 'amb': []})
time_values = OrderedDict({'ideal': [], 'amb': []})

for bundle_dir in bundle_dirs: # ideal and ambipolar
    print(bundle_dir[0].split('/')[-3])
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
                    #print(row[0])
                    continue

                #print(snap, str(row[0]))

        #print(snap_values[case][-1], time_values[case][-1], np.log10(peak_den[case][-1]))                    
        # Convert lists to numpy array


        data = np.load(snap_data, mmap_mode='r')
        column_density = data['column_density']
        radius_vector = data['positions']
        trajectory = data['trajectory']
        numb_densities = data['number_densities']
        magnetic_fields = data['magnetic_fields']
        threshold = data['thresholds']

        r_bundle, r_10, r_100, n_r = evaluate_reduction(magnetic_fields, numb_densities, threshold)

        readable = "{:02}:{:06.3f}".format(int((time.time()-start_time) // 60), (time.time()-start_time)  % 60)
        
        R10[case][snap]  =  R10[case].get(snap,  list(r_10*0)) + list(r_10)
        R100[case][snap] = R100[case].get(snap, list(r_100*0)) + list(r_100)
        NR[case][snap] = NR[case].get(snap, list(n_r*0))+ list(n_r)

        #        print("\nSnapshot: ", snap)
        #        print("Threshold 10 cm-3 yields mean(R10): ", np.mean(R10[case][snap]))
        #        print("Threshold 100cm-3 yields mean(R100): ", np.mean(R100[case][snap]))
        #        print("Elapsed time: ", readable, '\n')

#snap_values = np.array(snap_values)
#time_values = np.array(time_values)
#peak_den = np.array(peak_den)
#snap_values = np.array(snap_values)
#time_values = np.array(time_values)
#peak_den = np.array(peak_den)

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

for k, v in sorted(R10['ideal'].items()):
    mean_ideal.append(np.mean((np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])))
    median_ideal.append(np.median((np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])))
    ideal_snap.append(int(k))
    rdcut, x, mean, median, ten, s_size, f = statistics_reduction(np.array(R10['ideal'][k]), np.array(NR['ideal'][k])) 
    s_ideal.append((rdcut, x, mean, median, ten, s_size, k, f))
    fractions_i.append(float(f))
    Delta['ideal'][k] = (np.array(R10['ideal'][k]) - np.array(R100['ideal'][k])) / np.array(R100['ideal'][k])

for k, v in sorted(R10['amb'].items()):
    mean_amb.append(np.mean((np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])))
    median_amb.append(np.median((np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])))
    amb_snap.append(int(k))
    rdcut, x, mean, median, ten, s_size, f = statistics_reduction(np.array(R10['amb'][k]), np.array(NR['amb'][k])) 
    s_amb.append((rdcut, x, mean, median, ten, s_size, k, f))
    fractions_a.append(float(f))
    Delta['amb'][k] = (np.array(R10['amb'][k]) - np.array(R100['amb'][k])) / np.array(R100['amb'][k])


ideal_time = time_values['ideal']
mean_ir = [np.mean(r) for r in R100['ideal'].values()]
median_ir = [np.median(r) for r in R100['ideal'].values()]
amb_time = time_values['amb']
mean_ar = [np.mean(r) for r in R100['amb'].values()]
median_ar = [np.median(r) for r in R100['amb'].values()]

fig0, ax0 = plt.subplots()
ax0.plot(ideal_time, mean_ideal , label='mean $\\Delta_{ideal}$', linewidth=1.5, linestyle='-', color='darkorange')
ax0.plot(ideal_time, median_ideal, label='median $\\Delta_{ideal}$', linewidth=1.5, linestyle='--', color='darkorange')
ax0.plot(amb_time, mean_amb, label='mean $\\Delta_{amb}$', linewidth=1.5, linestyle='-', color='royalblue')
ax0.plot(amb_time, median_amb, label='median $\\Delta_{amb}$', linewidth=1.5, linestyle='--', color='royalblue')
ax0.set_ylabel('$(R_{10} - R_{100})/R_{100}$')
ax0.set_title('$\\Delta = (R_{10} - R_{100})/R_{100}$')
ax0.set_xlabel('Snapshots')
ax0.legend()
plt.savefig('./delta_threshold.png')
plt.show()

fig_ideal, ax_ideal = plt.subplots()
fig_amb, ax_amb = plt.subplots()


print(fractions_i)

ax_ideal.scatter(ideal_time, fractions_i, label='fractions', marker='x', color='black')
ax_ideal.plot(ideal_time, mean_ir, label='mean $\\Delta_{ideal}$', linewidth=1.5, linestyle='-', color='darkorange')
ax_ideal.plot(ideal_time, median_ir, label='median $\\Delta_{ideal}$', linewidth=1.5, linestyle='--', color='darkorange')
ax_ideal.set_ylabel('$R$')
ax_ideal.set_xlabel('time (Myrs)')
ax_ideal.legend()
ax_ideal.set_title("Ideal Case")



print(fractions_a)

ax_amb.scatter(amb_time, fractions_a, label='fractions', marker='x', color='black')
ax_amb.plot(amb_time, mean_ar, label='mean $\\Delta_{amb}$', linewidth=1.5, linestyle='-', color='royalblue')
ax_amb.plot(amb_time, median_ar, label='median $\\Delta_{amb}$', linewidth=1.5, linestyle='--', color='royalblue')
ax_amb.set_ylabel('$R$')
ax_amb.set_xlabel('time (Myrs)')
ax_amb.legend()
ax_amb.set_title("Ambipolar Diffusion Case")

fig_ideal.savefig('./time_reduction_ideal.png')
fig_amb.savefig('./time_reduction_amb.png')
plt.show()

for tup in s_ideal:
    rdcut, x, mean, median, ten, s_size, no, f = tup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(x, mean, label='mean', linewidth=1.5, linestyle='-', color='darkorange')
    ax1.plot(x, median, label='median', linewidth=1.5, linestyle='--', color='darkorange')
    ax1.plot(x, ten, label='10th percentile', linewidth=1.5, linestyle='-', color='royalblue')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$R$')
    ax1.set_xlabel('$n_g$')
    ax1.set_title(f'$f$ = {f}')
    ax1.legend()
    ax2.plot(x, s_size, label='sample size', linewidth=1.5, linestyle='-', color='darkorange')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'Sample size')
    ax2.set_xlabel('$n_g$')
    ax2.legend()
    plt.savefig(f'./reduction_density/ideal/ideal_{no}_reduction_density.png')
    plt.close()

for tup in s_amb:
    rdcut, x, mean, median, ten, s_size, no, f = tup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(x, mean , label='mean', linewidth=1.5, linestyle='-', color='darkorange')
    ax1.plot(x, median, label='median', linewidth=1.5, linestyle='--', color='darkorange')
    ax1.plot(x, ten, label='10th percentile', linewidth=1.5, linestyle='-', color='royalblue')
    ax1.set_xscale('log')
    ax1.set_ylabel('$R$')
    ax1.set_xlabel('$n_g$')
    ax1.set_title(f'$f$ = {f}')
    ax1.legend()
    ax2.plot(x, s_size, label='sample size', linewidth=1.5, linestyle='-', color='darkorange')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'Sample size')
    ax2.set_xlabel('$n_g$')
    ax2.legend()
    plt.savefig(f'./reduction_density/amb/amb_{no}_reduction_density.png')
    plt.close()
# to export RBundle into R10 and R100
# R10, R100 = zip(*RBundle)






