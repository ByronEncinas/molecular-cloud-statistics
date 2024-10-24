from collections import defaultdict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np
import random
import shutil
import h5py
import json
import sys
import os

from library import *

import time
    
rounds = int(sys.argv[1]) # number of files

group_one   = [] #dict()
group_two   = [] #dict()
group_three = [] #dict()
group_four  = []

for cycle in range(rounds):

    radius_vector  = np.array(np.load(f"arepo_npys/ArePositions{cycle}.npy", mmap_mode='r'))
    distance       = np.array(np.load(f"arepo_npys/ArepoTrajectory{cycle}.npy", mmap_mode='r'))
    bfield         = np.array(np.load(f"arepo_npys/ArepoMagneticFields{cycle}.npy", mmap_mode='r'))
    numb_density   = np.array(np.load(f"arepo_npys/ArepoNumberDensities{cycle}.npy", mmap_mode='r'))
    
    for index, B_i in enumerate(bfield):

        N_i = numb_density[index]

        if index != len(bfield) - 1:
            B_ip = bfield[index+1]
            N_ip = numb_density[index+1]
        else:
            B_ip = B_i
            N_ip = N_i

        B_error = np.abs(B_ip - B_i)/B_i # B_ip percentaje of error compared to previous line
        N_error = np.abs(N_ip - N_i)/N_i 

        #print([N_error, B_error])

        if N_i >= 10 and N_i <= 10e+2:
            group_one.append([N_error, B_error])
        elif N_i > 10e+2 and N_i <= 10e+4:
            group_two.append([N_error, B_error])
        elif N_i > 10e+4 and N_i <= 10e+6:
            group_three.append([N_error, B_error])
        elif N_i > 10e+6:
            group_four.append([N_error, B_error])

    groups = [group_one, group_two, group_three, group_four]

    # we'll make one histogram per group
    for j, g in enumerate(groups):
        if len(g) == 0:
            continue

        print(f"cycle: {cycle}: group {j+1}: lenght: ", len(g))

        g = np.array(g)

        eN = g[:,0]
        eB = g[:,1]

        if len(g) <= 10:
            bins = 10
        else:
            bins = len(g)//10

        # Create a figure and axes objects
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot histograms on the respective axes
        axs[0].hist(eN, bins=bins, color='skyblue', edgecolor='black')
        axs[0].set_title('Percentual difference Number density (%)')
        axs[0].set_xlabel('Bins')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(eB, bins=bins, color='skyblue', edgecolor='black')
        axs[1].set_title('Percentual difference Magnetic Field (%)')
        axs[1].set_xlabel('Bins')
        axs[1].set_ylabel('Frequency')

        axs[2].plot(bfield/np.max(bfield), color='red')
        axs[2].plot(numb_density/np.max(numb_density), color='blue')
        axs[2].set_title('Percentual difference Magnetic Field (%)')
        axs[2].set_xlabel('Bins')
        axs[2].set_ylabel('Frequency')

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        #plt.savefig("c_output_data/histogramdata={len(reduction_factor)}bins={bins}"+name+".png")
        plt.savefig(f"histograms/percentual_errors{cycle}-{j}.png")
