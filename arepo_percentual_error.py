import matplotlib.pyplot as plt
import numpy as np
import sys

from library import *

import time
    
rounds = int(sys.argv[1]) # number of files

group_names = ['Group 1 (0 to 10e+2)', 'Group 2 (10e+2 to 10e+4)', 
                'Group 3 (10e+4 to 10e+6)', 'Group 4 (> 10e+6)']

for cycle in range(rounds):
    step = '0.8'

    radius_vector  = np.load(f"arepo_npys/stepsizetest/{step}/ArePositions{cycle}.npy", mmap_mode='r')
    distance       = np.load(f"arepo_npys/stepsizetest/{step}/ArepoTrajectory{cycle}.npy", mmap_mode='r')
    bfield         = np.load(f"arepo_npys/stepsizetest/{step}/ArepoMagneticFields{cycle}.npy", mmap_mode='r')
    numb_density   = np.load(f"arepo_npys/stepsizetest/{step}/ArepoNumberDensities{cycle}.npy", mmap_mode='r')

    group_one   = [] 
    group_two   = [] 
    group_three = [] 
    group_four  = []

    for index, B_i in enumerate(bfield):

        N_i = numb_density[index]

        if index != len(bfield) - 1:
            B_ip = bfield[index+1]
            N_ip = numb_density[index+1]
        else:
            B_ip = B_i
            N_ip = N_i

        B_error = np.abs(B_ip - B_i) / B_i if B_i != 0 else 0
        N_error = np.abs(N_ip - N_i) / N_i if N_i != 0 else 0

        if 0 <= N_i <= 10e+2:
            group_one.append([N_error, B_error])
        elif 10e+2 < N_i <= 10e+4:
            group_two.append([N_error, B_error])
        elif 10e+4 < N_i <= 10e+6:
            group_three.append([N_error, B_error])
        elif N_i > 10e+6:
            group_four.append([N_error, B_error])

    groups = [group_one, group_two, group_three, group_four]

    for j, g in enumerate(groups):
        if len(g) == 0:
            continue

        print(f"cycle: {cycle}: group {j+1}: length: ", len(g))
        lenbfield = len(bfield)
        print("number of elements", lenbfield)

        g = np.array(g)

        eN = g[:,0]*100
        eB = g[:,1]*100

        bins = max(10, len(g) // 10)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].hist(eN, bins=bins, color='skyblue', edgecolor='black', label=group_names[j])
        axs[0].set_title('Percentual difference Number density (%)')
        axs[0].set_xlabel('Percent Difference')
        axs[0].set_ylabel('Frequency')
        axs[0].legend() 

        axs[1].hist(eB, bins=bins, color='skyblue', edgecolor='black' , label=group_names[j])
        axs[1].set_title('Percentual difference Magnetic Field (%)')
        axs[1].set_xlabel('Percent Difference')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()

        axs[2].plot(numb_density/np.max(numb_density), color='blue',label='Number Density')
        axs[2].plot(bfield/np.max(bfield), color='red',label='Magnetic Field')
        axs[2].set_title('Profile Shape N/cm^3 (blue) & B(s) (red)')
        axs[2].set_xlabel('Steps')
        axs[2].set_ylabel('Arbitrary Units')
        axs[2].legend()

        plt.tight_layout()

        plt.savefig(f"arepo_npys/stepsizetest/{step}/percentual_errors{cycle}-{j+1}.png")

