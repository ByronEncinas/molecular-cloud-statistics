from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from library import *
import os
import h5py
import json
import sys
import time
import csv
import glob

start_time = time.time()


"""  
Analysis of reduction factor

$$N(s) 1 - \sqrt{1-B(s)/B_l}$$

Parameters

Attribute: UnitLength_in_cm = 3.086e+18
Attribute: UnitMass_in_g = 1.99e+33
Attribute: UnitVelocity_in_cm_per_s = 100000.0

Name: PartType0/Coordinates
    Attribute: to_cgs = 3.086e+18
Name: PartType0/Density
    Attribute: to_cgs = 6.771194847794873e-23
Name: PartType0/Masses
    Attribute: to_cgs = 1.99e+33
Name: PartType0/Velocities
    Attribute: to_cgs = 100_000.0
Name: PartType0/MagneticField
"""

FloatType = np.float64
IntType = np.int32

if len(sys.argv)>4:
    #python3 arepo_reduction_factor_colors.py N rloc max_cycles num_file SLURM_JOB_ID
    #python3 arepo_reduction_factor_colors.py 500 0.5 100 430 
	N=int(sys.argv[1])
	rloc_boundary=float(sys.argv[2])
	max_cycles   =int(sys.argv[3])
	typpe = f'{sys.argv[4]}'
	num_file = f'{sys.argv[5]}'
	if len(sys.argv) < 6:
		sys.argv.append('NO_ID')
else:
    N            =5_000
    rloc_boundary=1   # rloc_boundary for boundary region of the cloud
    max_cycles   =500
    typpe = 'ideal'
    num_file = '430'

print(*sys.argv)


cycle = 0 
reduction_factor_at_numb_density = defaultdict()
reduction_factor = []

"""  B. Jesus Velazquez """

if typpe == 'ideal':
    subdirectory = 'ideal_mhd'
elif typpe == 'amb':
    subdirectory = 'ambipolar_diffusion'
else:
    subdirectory= ''

trajectory_path = f'cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt'


# Path to the input file
file_path = f'cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt'

# Lists to store column data
snap = []
time_value = []

# Open the file and read it using the CSV module
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        if num_file == str(row[0]):
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])
            snap =str(row[0])
            time_value = float(row[1])
            peak_den =  float(row[5])

print(Center)

# Convert lists to numpy arrays
snap_array = np.array(snap)
time_value_array = np.array(time_value)

# Get the list of files from the directory
directory_path = f"arepo_data/{subdirectory}"
file_list = glob.glob(f"{directory_path}/*.hdf5")

# Print the first 5 files for debugging/inspection
print(file_list[:5])

filename = 'arepo_data/snap_430.hdf5'

for f in file_list:
    if num_file in f:
        filename = f

data = h5py.File(filename, 'r')
# Access the 'Header' group

header_group = data['Header']
parent_folder = "cloud_tracker_slices/"+ typpe 
children_folder = os.path.join(parent_folder, 'ct_'+snap)
print(children_folder)
os.makedirs(children_folder, exist_ok=True)
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Volume   = Mass/Density

# Initialize gradients
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))

# Print the time
print(f"Time: {time_value} Myr")

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

#Center = Pos[np.argmax(Density),:]
CloudCord = Center.copy()

# ideal/amb #snap #ocationOf

# all positions are relative to the 'Center'
VoronoiPos-=Center
Pos-=Center

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize
xPosFromCenter = Pos[:,1]
Pos[xPosFromCenter > Boxsize/2,1]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,1] -= Boxsize
xPosFromCenter = Pos[:,2]
Pos[xPosFromCenter > Boxsize/2,2]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,2] -= Boxsize

# obtain width of cloud to set up radius
# calculate the a gaussian for the
densthresh = 100
rloc_boundary = 1 # average size is two parsec, allow them to be a little big bigger before rejection sampling

def get_along_lines(x_init=None, densthresh = 100):

    dx = 0.5

    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))
    volumes   = np.zeros((N+1,m))
    threshold = np.zeros((m,)).astype(int) # one value for each

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
    volumes_rev   = np.zeros((N+1,m))
    threshold_rev = np.zeros((m,)).astype(int) # one value for each

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init

    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3

    k=0

    mask = dens > densthresh # True if not finished
    un_masked = np.logical_not(mask)

    while np.any(mask):

        # Create a mask for values that are 10^2 N/cm^3 above the threshold
        mask = dens > densthresh # 1 if not finished
        un_masked = np.logical_not(mask) # 1 if finished

        aux = x[un_masked]

        x, bfield, dens, vol = Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        if len(threshold[un_masked]) != 0:
            unique_unmasked_max_threshold = np.max(np.unique(threshold[un_masked]))
            max_threshold = np.max(threshold)
        else:
            unique_unmasked_max_threshold = np.max(threshold)
            max_threshold = np.max(threshold)
        
        x[un_masked] = aux
        print(np.log10(dens[:3]))
        
        #print(threshold)
        # print(max_threshold, unique_unmasked_max_threshold)

        line[k+1,:,:]    = x
        volumes[k+1,:]   = vol
        bfields[k+1,:]   = bfield
        densities[k+1,:] = dens

        step_diff = max_threshold-unique_unmasked_max_threshold
        
        order_clause = step_diff >= 1_000
        percentage_clause = np.sum(un_masked)/len(mask) > 0.8

        if np.all(un_masked) or (order_clause and percentage_clause): 
            if (order_clause and percentage_clause):
                with open(f'isolated_radius_vectors{snap}.dat', 'a') as file: 
                    file.write(f"{order_clause} and {percentage_clause} of file {filename}\n")
                    file.write(f"{x_init[mask]}\n")
                print("80% of lines have concluded ")
            else:
                print("All values are False: means all crossed the threshold")
            break    

        k += 1
    
    threshold = threshold.astype(int)
    
    x = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]

    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    print(line_rev.shape)

    k=0

    mask_rev = dens > densthresh
    un_masked_rev = np.logical_not(mask_rev)
    
    while np.any((mask_rev)):

        mask_rev = dens > densthresh
        un_masked_rev = np.logical_not(mask_rev)

        aux = x[un_masked_rev]

        x, bfield, dens, vol = Heun_step(x, -dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold_rev += mask_rev.astype(int)

        if len(threshold_rev[un_masked_rev]) != 0:
            unique_unmasked_max_threshold = np.max(np.unique(threshold_rev[un_masked_rev]))
            max_threshold = np.max(threshold_rev)
        else:
            unique_unmasked_max_threshold = np.max(threshold_rev)
            max_threshold = np.max(threshold_rev)

        #print(max_threshold, unique_unmasked_max_threshold)
        print(np.log10(dens[:3]))
        x[un_masked_rev] = aux

        line_rev[k+1,:,:] = x
        volumes_rev[k+1,:] = vol
        bfields_rev[k+1,:] = bfield
        densities_rev[k+1,:] = dens 
                    
        step_diff = max_threshold-unique_unmasked_max_threshold
        
        order_clause = step_diff >= 1_000
        percentage_clause = np.sum(un_masked_rev)/len(mask_rev) > 0.8

        if np.all(un_masked_rev) or (order_clause and percentage_clause):
            if (order_clause and percentage_clause):
                with open(f'isolated_radius_vectors{snap}.dat', 'a') as file: 
                    file.write(f"{order_clause} and {percentage_clause} of file {filename}\n")
                    file.write(f"{x_init[mask_rev]}\n")
                print("80% of lines have concluded ")
            else:
                print("All values are False: means all crossed the threshold")
            break

        k += 1

    updated_mask = np.logical_not(np.logical_and(mask, mask_rev))
    
    threshold = threshold[updated_mask].astype(int)
    threshold_rev = threshold_rev[updated_mask].astype(int)

    # Apply updated_mask to the second axis of (N+1, m, 3) or (N+1, m) arrays
    line = line[:, updated_mask, :]  # Mask applied to the second dimension (m)
    volumes = volumes[:, updated_mask]  # Assuming volumes has shape (m,)
    bfields = bfields[:, updated_mask]  # Mask applied to second dimension (m)
    densities = densities[:, updated_mask]  # Mask applied to second dimension (m)

    # Apply to the reverse arrays in the same way
    line_rev = line_rev[:, updated_mask, :]
    volumes_rev = volumes_rev[:, updated_mask]
    bfields_rev = bfields_rev[:, updated_mask]
    densities_rev = densities_rev[:, updated_mask]
    
    radius_vector = np.append(line_rev[::-1, :, :], line, axis=0)
    magnetic_fields = np.append(bfields_rev[::-1, :], bfields, axis=0)
    numb_densities = np.append(densities_rev[::-1, :], densities, axis=0)
    volumes_all = np.append(volumes_rev[::-1, :], volumes, axis=0)

    #gas_densities   *= 1.0* 6.771194847794873e-23                      # M_sol/pc^3 to gram/cm^3
    #numb_densities   = gas_densities.copy() * 6.02214076e+23 / 1.00794 # from gram/cm^3 to Nucleus/cm^3

    # Initialize trajectory and radius_to_origin with the same shape
    trajectory = np.zeros_like(magnetic_fields)
    radius_to_origin = np.zeros_like(magnetic_fields)

    print("Magnetic fields shape:", magnetic_fields.shape)
    print("Radius vector shape:", radius_vector.shape)
    print("Numb densities shape:", numb_densities.shape)

    m = magnetic_fields.shape[1]
    print("Surviving lines: ", m, "out of: ", max_cycles)
	
    for _n in range(m): # Iterate over the first dimension
        prev = radius_vector[0, _n, :]
        for k in range(magnetic_fields.shape[0]):  # Iterate over the first dimension
            radius_to_origin[k, _n] = magnitude(radius_vector[k, _n, :])
            cur = radius_vector[k, _n, :]
            diff_rj_ri = magnitude(cur, prev)
            trajectory[k,_n] = trajectory[k-1,_n] + diff_rj_ri            
            prev = radius_vector[k, _n, :]
    
    trajectory[0,:]  = 0.0

    radius_vector   *= 1.0* 3.086e+18                                # from Parsec to cm
    trajectory      *= 1.0* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)
    volumes_all     *= 1.0#/(3.086e+18**3) 

    return bfields[0,:], radius_vector, trajectory, magnetic_fields, numb_densities, volumes_all, radius_to_origin, [threshold, threshold_rev]

def generate_vectors_in_core(max_cycles, densthresh, rloc=2.5):
    from scipy.spatial import cKDTree

    valid_vectors = []
    
    # Build a KDTree for nearest neighbor search
    tree = cKDTree(Pos)  # Pos contains Voronoi cell positions
    
    while len(valid_vectors) < max_cycles:
        points = np.random.uniform(low=-rloc, high=rloc, size=(max_cycles, 3))
        distances = np.linalg.norm(points, axis=1)
        
        # Keep only points inside the sphere
        inside_sphere = points[distances <= rloc]

        # Find the nearest Voronoi cell for each point
        _, nearest_indices = tree.query(inside_sphere)

        # Get the densities of the corresponding Voronoi cells
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > densthresh
        valid_points = inside_sphere[valid_mask]

        valid_vectors.extend(valid_points)

    return np.array(valid_vectors[:max_cycles])

x_init = generate_vectors_in_core(max_cycles, densthresh)

generated_points = generate_vectors_in_core(max_cycles, densthresh)
x, y = generated_points[:, 0], generated_points[:, 1]
#vor_x, vor_y = Pos[:, 0], Pos[:, 1]

# Create three subplots for X-Y, X-Z, and Y-Z projections
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# X-Y Projection
#axes[0].scatter(vor_x, vor_y, s=10, color='gray', alpha=0.5, label="Voronoi Cells")
axes[0].scatter(x, y, s=15, color='red', alpha=0.7, label="Generated Points")
axes[0].set_xlabel("X Coordinate")
axes[0].set_ylabel("Y Coordinate")
axes[0].set_title("X-Y Projection")

# X-Z Projection
x, z = generated_points[:, 0], generated_points[:, 2]
#vor_z = Pos[:, 2]
#axes[1].scatter(vor_x, vor_z, s=10, color='gray', alpha=0.5, label="Voronoi Cells")
axes[1].scatter(x, z, s=15, color='red', alpha=0.7, label="Generated Points")
axes[1].set_xlabel("X Coordinate")
axes[1].set_ylabel("Z Coordinate")
axes[1].set_title("X-Z Projection")

# Y-Z Projection
y, z = generated_points[:, 1], generated_points[:, 2]
#vor_y = Pos[:, 1]
#axes[2].scatter(vor_y, vor_z, s=10, color='gray', alpha=0.5, label="Voronoi Cells")
axes[2].scatter(y, z, s=15, color='red', alpha=0.7, label="Generated Points")
axes[2].set_xlabel("Y Coordinate")
axes[2].set_ylabel("Z Coordinate")
axes[2].set_title("Y-Z Projection")

# Adjust layout and show plots
for ax in axes:
    ax.legend()
    ax.axis("equal")
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(parent_folder, f'./x_init_above_thresh{densthresh}.png'))
plt.close(fig)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot generated points in red
ax.scatter(generated_points[:, 0], generated_points[:, 1], generated_points[:, 2], 
           s=15, color='red', alpha=0.7, label="Generated Points")

# Set axis limits to form a cube with side length 2.5
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([-2.5, 2.5])

# Labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.set_title("3D Distribution of Generated Vectors in Core")

# Legend and aspect ratio
ax.legend()
ax.view_init(elev=30, azim=45)  # Adjust view angle for better visualization
plt.savefig(f'./3D_plot_at_{densthresh}.png')
plt.close(fig)

print("Cores Used          : ", os.cpu_count())
print("Steps in Simulation : ", 2*N)
print("rloc                : ", rloc_boundary)
print("x_init              : ", x_init)
print("max_cycles          : ", max_cycles)
print("Boxsize             : ", Boxsize) # 256
print("Center              : ", Center) # 256
print("Posit Max Density   : ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume     : ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume     : ", Volume[np.argmax(Volume)]) # 256
print(f"Smallest Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmax(Volume)]}")
print(f"Biggest  Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmin(Volume)]}")

__, radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th = get_along_lines(x_init, densthresh)

print("Elapsed Time: ", (time.time() - start_time)/60.)

# Create the new arepo_npys directory
os.makedirs(children_folder, exist_ok=True)

# flow control to repeat calculations in no peak situations

m = magnetic_fields.shape[1]

threshold, threshold_rev = th

reduction_factor = list()
numb_density_at  = list()

min_den_cycle = list()

pos_red = dict()

for cycle in range(max_cycles):

    _from = N+1 - threshold_rev[cycle]
    _to   = N+1 + threshold[cycle]
    p_r = N + 1 - _from

    bfield    = magnetic_fields[_from:_to,cycle]
    distance = trajectory[_from:_to,cycle]
    numb_density = numb_densities[_from:_to,cycle]
    tupi = f"{x_init[cycle,0]},{x_init[cycle,1]},{x_init[cycle,2]}"

    bfield = bfield[1:]
    ds = np.diff(distance) 
    adaptive_sigma = 3*ds/np.mean(ds) #(ds > np.mean(ds))
    bfield = np.array([gaussian_filter1d(bfield, sigma=s)[i] for i, s in enumerate(adaptive_sigma)])

    distance = distance[1:]
    numb_density = numb_density[1:]

    #index_peaks, global_info = pocket_finder(bfield) # this plots
    pocket, global_info = pocket_finder(bfield, cycle, plot=False) # this plots
    index_pocket, field_pocket = pocket[0], pocket[1]

    min_den_cycle.append(min(numb_density))
    
    globalmax_index = global_info[0]
    globalmax_field = global_info[1]

    x_r = distance[p_r]
    B_r = bfield[p_r]
    n_r = numb_density[p_r]

    # finds index at which to insert p_r and be kept sorted
    p_i = find_insertion_point(index_pocket, p_r)

    print("random index: ", p_r, "assoc. B(s_r), n_g(s_r):",B_r, n_r, "peak's index: ", index_pocket)
    
    """How to find index of Bl?"""

    print("Maxima Values related to pockets: ", len(index_pocket), p_i)

    try:
        # possible error is len(index_pocket) is only one or two elements
        closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
        B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
        B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
        success = True  # Flag to track if try was successful

    except:
        R = 1
        reduction_factor.append(R)
        numb_density_at.append(n_r)
        pos_red[tupi] = R
        success = False  # Set flag to False if there's an exception
        continue

    # Only execute this block if try was successful
    if success:
        if B_r / B_l < 1:
            R = 1 - np.sqrt(1 - B_r / B_l)
            reduction_factor.append(R)
            numb_density_at.append(n_r)
            pos_red[tupi] = R
        else:
            R = 1
            reduction_factor.append(R)
            numb_density_at.append(n_r)
            pos_red[tupi] = R

    print("Closest local maxima 'p':", closest_values)
    print("Bs: ", B_r, "ns: ", n_r)
    print("Bi: ", bfield[closest_values[0]], "Bj: ", bfield[closest_values[1]])

    if B_r/B_l < 1:
        print(" B_r/B_l =", B_r/B_l, "< 1 ") 
    else:
        print(" B_r/B_l =", B_r/B_l, "> 1 so CRs are not affected => R = 1") 

from collections import Counter

counter = Counter(reduction_factor)

pos_red = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pos_red.items()}

with open(os.path.join(children_folder, 'PARAMETER_reduction'), 'w') as file:
    file.write(f"{filename}\n")
    file.write(f"{peak_den}\n")
    file.write(f"Cores Used: {os.cpu_count()}\n")
    file.write(f"Snap Time (Myr): {time_value}\n")
    file.write(f"rloc (Pc) : {rloc_boundary}\n")
    file.write(f"x_init (Pc)        :\n {x_init}\n")
    file.write(f"max_cycles         : {max_cycles}\n")
    file.write(f"Boxsize (Pc)       : {Boxsize} Pc\n")
    file.write(f"Center (Pc, Pc, Pc): {CloudCord[0]}, {CloudCord[1]}, {CloudCord[2]} \n")
    file.write(f"Posit Max Density (Pc, Pc, Pc): {Pos[np.argmax(Density), :]}\n")
    file.write(f"Smallest Volume (Pc^3)   : {Volume[np.argmin(Volume)]} \n")
    file.write(f"Biggest  Volume (Pc^3)   : {Volume[np.argmax(Volume)]}\n")
    file.write(f"Smallest Density (M☉/Pc^3)  : {Density[np.argmax(Volume)]} \n")
    file.write(f"Biggest  Density (M☉/Pc^3) : {Density[np.argmin(Volume)]}\n")
    file.write(f"Smallest Density (N/cm^3)  : {Density[np.argmax(Volume)]*gr_cm3_to_nuclei_cm3} \n")
    file.write(f"Biggest  Density (N/cm^3) : {Density[np.argmin(Volume)]*gr_cm3_to_nuclei_cm3}\n")
    file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")

# Print elapsed time
print(f"Elapsed time: {(time.time() - start_time)/60.} Minutes")

# Specify the file path
file_path = os.path.join(children_folder, f'reduction_factor{sys.argv[-1]}.json')

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(reduction_factor, json_file)

# Specify the file path
file_path = os.path.join(children_folder,f'numb_density{sys.argv[-1]}.json')

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(numb_density_at, json_file)

# Specify the file path
file_path = os.path.join(children_folder,f'position_vector{sys.argv[-1]}')

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(pos_red, json_file) # [x,y,z] = R basicly a 3D stochastic functin

"""# Graphs"""

#plot_trajectory_versus_magnitude(trajectory, magnetic_fields, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])

bins=len(reduction_factor)//10 

if bins == 0:
    bins=1

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
plt.savefig(os.path.join(children_folder,f"{typpe}_hist={len(reduction_factor)}bins={bins}.png"))

if True:

    # Extract data from the dictionary
    x = np.log10(numb_density_at)   # log10(numb number density)
    y = np.array(reduction_factor)              # reduction factor R

    # Plot original scatter plot
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    axs.scatter(x, y, marker="x", s=5, color='red', label='Data points')
    axs.set_title('Histogram of Reduction Factor (R)')
    axs.set_ylabel('$(R)$')
    axs.set_xlabel('$log_{10}(n_g ($N/cm^{-3}$))$ ')

    # Compute binned statistics
    num_bins = 100

    # Median binned statistics
    bin_medians, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='median', bins=num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    axs.plot(bin_centers, bin_medians, marker="+", color='#17becf', linestyle='-', label='Binned medians')

    # Mean binned statistics
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)
    axs.plot(bin_centers, bin_means, marker="x", color='pink', linestyle='-', label='Binned means')

    # Overall mean and median
    overall_mean = np.average(y)
    overall_median = np.median(y)

    mean = np.ones_like(y) * overall_mean
    median = np.ones_like(y) * overall_median

    axs.plot(x, mean, color='dimgrey', linestyle='--', label=f'Overall mean ({overall_mean:.2f})')
    axs.plot(x, median, color='dimgray', linestyle='--', label=f'Overall median ({overall_median:.2f})')

    # Add legend
    axs.legend()

    plt.savefig(os.path.join(children_folder,f"mean_median.png"))
    plt.close(fig)
    #plt.show()

    # Define the number of bins
    num_bins = 100

    # Compute binned statistics
    bin_medians, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='median', bins=num_bins)
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=num_bins)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create the figure and axis
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    # Plot the histograms using Matplotlib
    axs.hist(bin_edges[:-1], bins=bin_edges, weights=bin_medians, alpha=0.5, label='medians', color='c', edgecolor='darkcyan')
    axs.hist(bin_edges[:-1], bins=bin_edges, weights=-bin_means, alpha=0.5, label='means', color='m', edgecolor='darkmagenta')

    # Set the labels and title
    axs.set_title('Histograms of Binned Medians and Means (Inverted)')
    axs.set_ylabel('$(R)$')
    axs.set_xlabel('$log_{10}(n_g ($N/cm^{-3}$))$ ')

    # Add legend
    axs.legend(loc='center')

    # save figure
    plt.savefig(os.path.join(children_folder,f"mirrored_histograms.png"))

    # Show the plot
    plt.close(fig)
    #plt.show()