from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import sys, time, glob, re
import warnings, csv

start_time = time.time()
pc_to_cm = 3.086 * 10e+18  # cm per parsec


amb_bundle   = sorted(glob.glob('./thesis_los/N/amb/*/DataBundle*.npz'))
ideal_bundle = sorted(glob.glob('./thesis_los/N/ideal/*/DataBundle*.npz'))

bundle_dirs = [ideal_bundle, amb_bundle]

readable = "{:02}:{:06.3f}".format(int((time.time() - start_time) // 60), (time.time() - start_time) % 60)

def extract_number(filename):
    match = re.search(r'(\d+)(?=\_.json)', filename)
    return int(match.group(0)) if match else 0

peak_den    = OrderedDict({'ideal': [], 'amb': []})
snap_values = OrderedDict({'ideal': [], 'amb': []})
time_values = OrderedDict({'ideal': [], 'amb': []})
CD = OrderedDict({'ideal': OrderedDict(), 'amb': OrderedDict()})

# It seems this line was repeated
# peak_den = OrderedDict(...) etc. already done above

# Missing: Initialize CD — this must be declared first!
# CD = OrderedDict({'ideal': {}, 'amb': {}})

ideal_mhd_times = {
    0.125, 0.25, 0.375, 0.4875, 0.575, 0.65, 0.775, 0.8625,
    0.9625, 1.0875, 1.1625, 1.2875, 1.4, 1.525, 1.64375, 1.70625,
    1.78125, 1.9, 2.0125, 2.1375, 2.2625, 2.5125, 2.6375, 2.7625,
    2.8875, 3.0125, 3.1375, 3.2625, 3.375, 3.5, 3.625, 3.75,
    3.875, 4.0, 4.125, 4.25, 4.375
}
ambipolar_diffusion_times = {
    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
    1.125, 1.25, 1.375, 1.5, 1.5875, 1.7125, 1.8375, 1.9625,
    2.0875, 2.2125, 2.3375, 2.725, 2.95, 3.2, 3.45, 3.7, 3.95,
    4.2, 4.286719, 4.290137, 4.29035, 4.290399, 4.290417, 4.290426,
    4.290433, 4.290437, 4.290441, 4.290445, 4.290448, 4.290452,
    4.290456, 4.29046, 4.290464, 4.290467, 4.290471, 4.290475,
    4.290479, 4.290483, 4.290487, 4.29049, 4.290492
}


for bundle_dir in bundle_dirs:  # ideal and ambipolar

    repeated = set()
    #print(bundle_dir)
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
                    if np.round(float(row[1]), 6) in ideal_mhd_times and case == 'ideal':
                        repeated.add(snap)
                        snap_values[case].append(str(row[0]))
                        time_values[case].append(float(row[1]))
                        peak_den[case].append(float(row[-1]))
                        continue

                    if np.round(float(row[1]), 6) in ambipolar_diffusion_times and case == 'amb':
                        repeated.add(snap)
                        snap_values[case].append(str(row[0]))
                        time_values[case].append(float(row[1]))
                        peak_den[case].append(float(row[-1]))
                        continue


        data = np.load(snap_data, mmap_mode='r')
        threshold = data['thresholds']
        threshold_rev = data['thresholds_rev']        
        column_density = data['column_densities']*pc_to_cm
        radius_vector = data['positions']
        numb_densities = data['number_densities']

        N = column_density.shape[0]
        snap_columns_sliced = []
        for i in range(column_density.shape[1]):
            snap_columns_sliced += [np.max(column_density[:, i])]
        
        CD[case][snap] = CD[case].get(snap, snap_columns_sliced * 0) + snap_columns_sliced

ideal_time = time_values['ideal']
amb_time = time_values['amb']

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
ax.set_title('Column Density along line of sight (ideal)')
ax.set_yscale('log')
ax.grid(True)
ax.set_xticklabels(round_time, rotation=60)
ax.locator_params(axis='x', nbins=15)
plt.tight_layout()
plt.savefig("./los_cd_ideal.png")
plt.close()


median = np.array([np.median(arr) for arr in data])
mean = np.array([np.mean(arr) for arr in data])
lower_68 = np.array([np.percentile(arr, 16) for arr in data])
upper_68 = np.array([np.percentile(arr, 84) for arr in data])
lower_95 = np.array([np.percentile(arr, 2.5) for arr in data])
upper_95 = np.array([np.percentile(arr, 97.5) for arr in data])

# X-axis: snapshot indices
fig, ax = plt.subplots()

ax.fill_between(round_time, lower_95, upper_95, color='pink', label='Interpecentile range (2.5th-97.5th)')

ax.fill_between(round_time, lower_68, upper_68, color='red', alpha=0.6, label='Interpecentile range (16th-84th)')

ax.plot(round_time, median, color='black',linestyle='--', linewidth=1, label='Median')
ax.plot(round_time, mean, color='black',linestyle='-', linewidth=1, label='Mean')

ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title(f'Column Density along line of sight (ideal)')
ax.set_yscale('log')
ax.set_xticks(round_time)
ax.set_xticklabels(round_time, rotation=60)
ax.locator_params(axis='x', nbins=10)
ax.grid(True)
ax.legend(loc='upper left', frameon=True, fontsize=11)

plt.tight_layout()
plt.savefig(f"./los_cd_ideal_inter.png")
plt.close()


# --- AMB CASE ---
labels = list(CD['amb'].keys())  # snapshot labels
data = list(CD['amb'].values())  # column density arrays

round_time = [np.round(t, 6) for t in amb_time]
for t in round_time:
    print("Amb: ", t)
fig, ax = plt.subplots()
ax.boxplot(data, flierprops=dict(marker='|', markersize=2, color='red'))
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title('Column Density along line of sight (non-ideal)')
ax.set_yscale('log')
ax.grid(True)
ax.set_xticklabels(round_time, rotation=60)
ax.locator_params(axis='x', nbins=15)
plt.tight_layout()
plt.savefig("./los_cd_amb.png")
plt.close()

median = np.array([np.median(arr) for arr in data])
mean = np.array([np.mean(arr) for arr in data])
lower_68 = np.array([np.percentile(arr, 16) for arr in data])
upper_68 = np.array([np.percentile(arr, 84) for arr in data])
lower_95 = np.array([np.percentile(arr, 2.5) for arr in data])
upper_95 = np.array([np.percentile(arr, 97.5) for arr in data])

# X-axis: snapshot indices
x = np.arange(len(labels))

fig, ax = plt.subplots()

ax.fill_between(round_time, lower_95, upper_95, color='pink', label='Interpecentile range (2.5th-97.5th)')
ax.fill_between(round_time, lower_68, upper_68, color='red', alpha=0.6, label='Interpecentile range (16th-84th)')
ax.plot(round_time, median, color='black',linestyle='--', linewidth=1, label='Median')
ax.plot(round_time, mean, color='black',linestyle='-', linewidth=1, label='Mean')
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Time (Myrs)')
ax.set_title(f'Column Density along line of sight (non-ideal)')
ax.set_yscale('log')
ax.set_xticks(round_time)
ax.set_xticklabels(round_time, rotation=60)
ax.locator_params(axis='x', nbins=15)
ax.grid(True)
ax.legend(loc='upper left', frameon=True, fontsize=11)

plt.tight_layout()
plt.savefig(f"./los_cd_amb_inter.png")
plt.close()
