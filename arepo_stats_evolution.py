import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
import json
import re
from library import *

import time

start_time = time.time()


"""  
This code takes json files from different evolutionary stages in A. Mayer Simulations.

- Plots the evolution of the median, mean and mode; 10th, 5th percentile
    - Note that this data does not necesarily belong to the same cloud, its just used as a measure of the whole 256^3 box
- I want to know their positions in 3D space
"""

import glob

file_list = glob.glob('cluster_outputs/histCAS/1000/*/')

amb_files   = dict()
ideal_files = dict()
print(file_list)
for file in sorted(file_list):
    # Extract the number from the filename
    print(file)

    JSON_FILE_PATH = glob.glob(file + '*.json') #os.path.join(file, '*.json')
    
    if len(JSON_FILE_PATH) == 0:
        continue

    JSON_FILE_PATH = JSON_FILE_PATH[1]
    no = str(file[-4:-1])
    print(no)

    if 'ideal' in file:
        # if we enter this if statement, we will save for that file the mean, median and mode of R in this snap
        # Open and read the file line by line
        with open(file + 'output' , 'r') as file:
            for line in file:
                # Search for the line containing "Snap Time"
                match = re.search(r"Snap Time \(Myr\): ([\d.]+)", line)
                if match:
                    snap_time = float(match.group(1))  # Extract the value as a float

        with open(JSON_FILE_PATH, 'r') as JSONfile:
            # Load the JSON data into a Python list and append to reduction_factor
            data = list(json.load(JSONfile))

        # Calculate percentiles
        percentile_25 = np.percentile(data, 25)  # 25th percentile
        percentile_10 = np.percentile(data, 10)  # 10th percentile
        percentile_5  = np.percentile(data, 5)    # 5th percentile

        # Calculate median
        mean   = np.mean(data)
        median = np.median(data)

        # Calculate mode
        mode_result = stats.mode(data, keepdims=True)  # keepdims ensures compatibility with arrays
        mode = mode_result[0]  # The most frequent value
        mode_count = mode_result[1]  # Number of occurrences of the mode

        # Print results
        print("Time:", snap_time)
        print("Size:", len(data))
        print("Mean:", mean)
        print("Median:", median)
        print("Mode:", mode, "with count:", mode_count)
        print("25th Percentile:", percentile_25)
        print("10th Percentile:", percentile_10)
        print("5th Percentile:", percentile_5)
        ideal_files[no] = (mean, median, mode, percentile_25, percentile_10, percentile_5, len(data), snap_time)

    elif 'amb' in file:
                # Open and read the file line by line
        with open(file + 'output' , 'r') as file:
            for line in file:
                # Search for the line containing "Snap Time"
                match = re.search(r"Snap Time \(Myr\): ([\d.]+)", line)
                if match:
                    snap_time = float(match.group(1))  # Extract the value as a float
        # if we enter this if statement, we will save for that file the mean, median and mode of R in this snap
        with open(JSON_FILE_PATH, 'r') as JSONfile:
            # Load the JSON data into a Python list and append to reduction_factor
            data = list(json.load(JSONfile))

        # Calculate percentiles
        percentile_25 = np.percentile(data, 25)  # 25th percentile
        percentile_10 = np.percentile(data, 10)  # 10th percentile
        percentile_5 = np.percentile(data, 5)    # 5th percentile

        # Calculate median
        mean   = np.mean(data)
        median = np.median(data)

        # Calculate mode
        mode_result = stats.mode(data, keepdims=True)  # keepdims ensures compatibility with arrays
        mode = mode_result[0]  # The most frequent value
        mode_count = mode_result[1]  # Number of occurrences of the mode

        # Print results
        print("Size:", len(data))
        print("Mean:", mean)
        print("Median:", median)
        print("Mode:", mode, "with count:", mode_count)
        print("25th Percentile:", percentile_25)
        print("10th Percentile:", percentile_10)
        print("5th Percentile:", percentile_5)
        amb_files[no] = (mean, median, mode, percentile_25, percentile_10, percentile_5, len(data), snap_time)


# Flag to control whether to plot the second x-axis
plot_second_xaxis = True  # Set to False to hide the second x-axis

Legend = ['ideal', 'amb']

# Extracting data for the graph
for i, dict_files in enumerate([ideal_files, amb_files]):

    no = np.array(list(dict_files.keys()), dtype=int)  # x-axis values
    means = np.array([v[0] for v in dict_files.values()])
    medians = np.array([v[1] for v in dict_files.values()])
    percentile_25 = np.array([v[3] for v in dict_files.values()])
    percentile_10 = np.array([v[4] for v in dict_files.values()])
    percentile_5 = np.array([v[5] for v in dict_files.values()])
    sample_size = np.array([v[6] for v in dict_files.values()])
    times = np.round(np.array([v[7] for v in dict_files.values()], dtype=float), 5)


    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting on the first x-axis (no)
    ax1.plot(no, means, marker='o', color='blue', label='Mean')
    ax1.plot(no, medians, marker='s', color='orange', label='Median')
    ax1.plot(no, percentile_25, marker='D', color='red', label='25th Percentile')
    ax1.plot(no, percentile_10, marker='v', color='purple', label='10th Percentile')
    ax1.plot(no, percentile_5, marker='x', color='brown', label='5th Percentile')

    ax1.set_xlabel("Snapshot (as integer)", fontsize=12)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.set_title("Regional Average Time Evolution of R", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Conditionally add the second x-axis for "times"
    if plot_second_xaxis:
        ax2 = ax1.twiny()  # Create a second x-axis
        ax2.set_xlim(ax1.get_xlim())  # Match the limits of the first axis
        ax2.set_xlabel("Myrs", fontsize=12)
        ax2.set_xticks(no)  # Optionally match the ticks to `no`
        ax2.set_xticklabels(times)  # Use `times` as the labels for the second x-axis

    # Show the plot
    plt.tight_layout()

    plt.savefig(f'cluster_outputs/histCAS/RegionalAverageInTime_{Legend[i]}.jpeg', bbox_inches='tight')
    plt.show()
