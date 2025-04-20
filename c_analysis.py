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

for bundle_dir in bundle_dirs:  # ideal and ambipolar

    repeated = set()

    print(bundle_dir)
    if bundle_dir == []:
        continue
    case = str(bundle_dir[0].split('/')[-3])
    
    
    for snap_data in bundle_dir:  # from 000 to 490 
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
                    continue

        data = np.load(snap_data, mmap_mode='r')
        
        column_density = data['column_densities']*pc_to_cm
        radius_vector = data['positions']
        numb_densities = data['number_densities']
        threshold = data['thresholds']
        threshold_rev = data['thresholds_rev']

        snap_columns_sliced = []
        for column in enumerate(column_density[0,:]):
            snap_columns_sliced = []
            CD[case][float(row[1])] = CD[case].get(float(row[1]), list(column_density[-1, :].tolist() * 0)) + column_density[-1, :].tolist()
        #  Make sure CD[case] exists and is a dict
        #CD[case][snap] = CD[case].get(snap, list(column_density[-1, :].tolist() * 0)) + column_density[-1, :].tolist()
        
        print(np.mean(CD[case][float(row[1])]), "cm^-2")

labels = list(CD['ideal'].keys())  # snapshot labels
data = list(CD['ideal'].values())  # column density arrays

from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots()
ax.boxplot(data, flierprops=dict(marker='|', markersize=2, color='red'))
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Snapshots')
ax.set_title('Column Density along CR path (ideal)')
ax.set_yscale('log')
ax.grid(True)
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))  # max 10 x labels
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("./los_cd_ideal.png")
plt.show()

# --- AMB CASE ---
labels = list(CD['amb'].keys())  # snapshot labels
data = list(CD['amb'].values())  # column density arrays

fig, ax = plt.subplots()
ax.boxplot(data, flierprops=dict(marker='|', markersize=2, color='red'))
ax.set_ylabel('Effective Column Density')
ax.set_xlabel('Snapshots')
ax.set_title('Column Density along CR path (amb)')
ax.set_yscale('log')
ax.grid(True)
ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))  # max 10 x labels
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("./los_cd_amb.png")
plt.show()

