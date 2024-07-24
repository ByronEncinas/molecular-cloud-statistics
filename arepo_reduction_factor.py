from library import *
from collections import Counter, defaultdict
import random
import subprocess
import matplotlib.pyplot as plt
import json

"""  
Using Margo Data

Analysis of reduction factor

$$N(s) 1 - \sqrt{1-B(s)/B_l}$$

Where $B_l$ corresponds with (in region of randomly choosen point) the lowest between the highest peak at both left and right.
where $s$ is a random chosen point at original 128x128x128 grid.

1.- Randomly select a point in the 3D Grid. 
2.- Follow field lines until finding B_l, if non-existent then change point.
3.- Repeat 10k times
4.- Plot into a histogram.

contain results using at least 20 boxes that contain equally spaced intervals for the reduction factor.

# Calculating Histogram for Reduction Factor in Randomized Positions in the 128**3 Cube 

"""
# flow control to repeat calculations in no peak situations
cycle = 0 

import sys

if len(sys.argv) >= 2:
    max_cycles = int(sys.argv[1])
    print("max cycles:", max_cycles)
else:
    max_cycles = int(100)
    print("max cycles:", max_cycles)

max_cycles = int(max_cycles)

reduction_factor_at_gas_density = defaultdict()

number_of_points = "100"
rloc_boundary    = "80"

print(sys.argv)

reduction_factor = np.zeros(max_cycles)

if "-1" in sys.argv:
    # Assuming your JSON file is named 'data.json'
    file_path = 'random_distributed_reduction_factor.json'

    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data into a Python list
        reduction_factor = np.array(json.load(file))

    cycle == max_cycles
    

while (int(cycle) < int(max_cycles)):

    # Path to the Python file you want to execute
    file_to_run = 'arepo_get_field_lines.py'
    
    rloc_center      = str(random.uniform(0,1)*float(rloc_boundary)/4)

    print("rloc_center:= ", rloc_center)

    args =  [number_of_points, rloc_boundary, rloc_center, "postprocess"]
    
# Running the file (for Python 3.6 compatibility)
    result = subprocess.run(['python3', file_to_run] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

    # Accessing the output and errors
    output = result.stdout
    errors = result.stderr

    print("Output:", output)
    print("Errors:", errors)

    distance     = np.array(np.load("arepo_output_data/ArepoTrajectory.npy", mmap_mode='r'))
    bfield       = np.array(np.load("arepo_output_data/ArepoMagneticFields.npy", mmap_mode='r'))
    numb_density = np.array(np.load("arepo_output_data/ArepoNumberDensities.npy", mmap_mode='r'))

    """# Obtained position along the field lines, now we find the pocket"""

    #index_peaks, global_info = pocket_finder(bfield) # this plots
    pocket, global_info = pocket_finder(bfield, cycle, plot=False) # this plots
    index_pocket, field_pocket = pocket[0], pocket[1]


    # we can evaluate reduction factor if there are no pockets
    if len(index_pocket) < 2:
        # it there a minimum value of peaks we can work with? yes, two
        continue

    globalmax_index = global_info[0]
    globalmax_field = global_info[1]

    # Calculate the range within the 80th percentile
    start_index = len(bfield) // 10  # Skip the first 10% of indices
    end_index = len(bfield) - start_index  # Skip the last 10% of indices

    # we gotta find peaks in the interval   (B_l < random_element < B_h)
    # Generate a random index within the range
    p_r = random.randint(index_pocket[0], index_pocket[-1])
    #s_r = distance[p_r]
    B_r = bfield[p_r]

    print("random index: ", p_r, "peak's index: ", index_pocket)
    
    """How to find index of Bl?"""

    # Bl it is definitely between two peaks, we need to verify is also inside a pocket
    # such that Bl < Bs < Bh (p_i < p_r < p_j)

    # finds index at which to insert p_r and be kept sorted
    p_i = find_insertion_point(index_pocket, p_r)

    #print()
    print("Random Index:", p_r, "assoc. B(s_r):",B_r)
    print("Maxima Values related to pockets: ",len(index_pocket), p_i)

    if p_i is not None:
        # If p_i is not None, select the values at indices p_i-1 and p_i
        closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
    else:
        # If p_i is None, select the two closest values based on some other criteria
        continue

    if len(closest_values) == 2:
        B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
        B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
    else:
        continue

    if B_r/B_l < 1:
        R = 1 - np.sqrt(1-B_r/B_l)
        reduction_factor[cycle] = R
        cycle += 1
        print("Bl: ", B_l, " B_r/B_l =", B_r/B_l, "< 1 ") 
    else:
        R = 1
        reduction_factor[cycle] = 1.
        cycle += 1
        print("Bl: ", B_l, " B_r/B_l =", B_r/B_l, "> 1 so CRs are not affected => R = 1") 
    
    print("Closest local maxima 'p':", closest_values)
    print("Bs: ", bfield[p_r], "Bi: ", bfield[closest_values[0]], "Bj: ", bfield[closest_values[1]])
    
    """
    bs: where bs is the field magnitude at the random point chosen 
    bl: magnetic at position s of the trajectory
    """
    print(cycle)
        

# Specify the file path
file_path = 'random_distributed_reduction_factor.json'

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(reduction_factor.tolist(), json_file)

"""# Graphs"""

#plot_trajectory_versus_magnitude(distance, bfield, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])

bins=len(reduction_factor)//10

inverse_reduction_factor = [1/reduction_factor[i] for i in range(len(reduction_factor))]

# try plt.stairs(*np.histogram(inverse_reduction_factor, 50), fill=True, color='skyblue')

# Create a figure and axes objects
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot histograms on the respective axes
axs[0].hist(reduction_factor, bins=bins, color='skyblue', edgecolor='black')
axs[0].set_yscale('log')
axs[0].set_title('Histogram of Reduction Factor (R)')
axs[0].set_xlabel('Bins')
axs[0].set_ylabel('Frequency')

axs[1].hist(inverse_reduction_factor, bins=bins, color='skyblue', edgecolor='black')
axs[1].set_yscale('log')
axs[1].set_title('Histogram of Inverse Reduction Factor (1/R)')
axs[1].set_xlabel('Bins')
axs[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Save the figure
#plt.savefig("c_output_data/histogramdata={len(reduction_factor)}bins={bins}"+name+".png")
plt.savefig("arepo_output_data/hist={len(reduction_factor)}bins={bins}.png")

# Show the plot
#plt.show()