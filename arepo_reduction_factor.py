from collections import defaultdict
from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
from scipy import spatial
import healpy as hp
import numpy as np
import random
import time
import h5py
import json
import sys
import os

from library import *

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

"""
Parameters

- [N] default is 50 as the total number of steps in the simulation
- [dx] default 4/N of the rloc_boundary (radius of spherical region of interest) variable

"""
FloatType = np.float64
IntType = np.int32

if len(sys.argv)>2:
	# first argument is a number related to rloc_boundary
	N=int(sys.argv[1])
	rloc_boundary=float(sys.argv[2])
	rloc_center  =int(sys.argv[3])
	max_cycles   =int(sys.argv[4])
else:
    N            =200
    rloc_boundary=256   # rloc_boundary for boundary region of the cloud
    rloc_center  =1     # rloc_boundary for inner region of the cloud
    max_cycles   =100


# flow control to repeat calculations in no peak situations
cycle = 0 

reduction_factor_at_gas_density = defaultdict()

reduction_factor = np.array([])

filename = 'arepo_data/snap_430.hdf5'

"""
Functions/Methods

- Data files provided do not contain  
- 
"""
def magnitude(new_vector, prev_vector=[0.0,0.0,0.0]): 
    return np.sqrt(sum([(new_vector[i]-prev_vector[i])*(new_vector[i]-prev_vector[i]) for i in range(len(new_vector))]))

def get_magnetic_field_at_points(x, Bfield, rel_pos):
	n = len(rel_pos[:,0])
	local_fields = np.zeros((n,3))
	for  i in range(n):
		local_fields[i,:] = Bfield[i,:]
	return local_fields

def get_density_at_points(x, Density, Density_grad, rel_pos):
	n = len(rel_pos[:,0])	
	local_densities = np.zeros(n)
	for  i in range(n):
		local_densities[i] = Density[i] + np.dot(Density_grad[i,:], rel_pos[i,:])
	return local_densities

def find_points_and_relative_positions(x, Pos):
	dist, cells = spatial.KDTree(Pos[:]).query(x, k=1,workers=12)
	rel_pos = VoronoiPos[cells] - x
	return dist, cells, rel_pos

def find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos)
	local_fields = get_magnetic_field_at_points(x, Bfield[cells], rel_pos)
	local_densities = get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos)
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	
	return local_fields, abs_local_fields, local_densities, cells
	
def Heun_step(x, dx, Bfield, Density, Density_grad, Pos):

	local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos)
	local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T
		
	if dx > 0:
		dx = 0.3*((4/3)*Volume[cells][0]/np.pi)**(1/3)
	else:
		dx = -0.3*((4/3)*Volume[cells][0]/np.pi)**(1/3)

	x_tilde = x + dx * local_fields_1
	local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos)
	local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
	abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1))

	unitary_local_field = (local_fields_1 + local_fields_2) / np.tile(abs_sum_local_fields, (3,1)).T
	x_final = x + 0.5 * dx * unitary_local_field[0]
	
	x_final = x + 0.5 * dx * (local_fields_1 + local_fields_2)
	
	return x_final, abs_local_fields_1, local_densities

"""  B. Jesus Velazquez One Dimensional """

data = h5py.File(filename, 'r')

print(filename, "Loaded")

Boxsize = data['Header'].attrs['BoxSize'] #

VoronoiPos = np.array(data['PartType0']['Coordinates'], dtype=FloatType) # Voronoi Point in Cell
Pos = np.array(data['PartType0']['CenterOfMass'], dtype=FloatType)  # CenterOfMass in Cell
Bfield = np.array(data['PartType0']['MagneticField'], dtype=FloatType)
Bfield_grad  = np.zeros((len(Bfield),3))
Density = np.array(data['PartType0']['Density'], dtype=FloatType)
Density_grad = np.zeros((len(Density),3))
Mass = np.array(data['PartType0']['Masses'], dtype=FloatType)
Volume = Mass/Density

# printing relevant info about the data
# 1 Parsec  = 3.086e+18 cm
# 1 Solar M = 1.9885e33 gr
# 1 Km/s    = 100000 cm/s

"""  
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

Bfield  *= 1.0#  (3.086e+18/1.9885e33)**(-1/2) # in cgs
Density *= 1.0# 6.771194847794873e-23          # in cgs
Mass    *= 1.0# 1.9885e33                      # in cgs
Volume   = Mass/Density

#Center= 0.5 * Boxsize * np.ones(3) # Center # Default

#Center = np.array( [91,       -110,          -64.5]) #117
#Center = np.array( [96.6062303,140.98704002, 195.78020632]) #117
Center = Pos[np.argmax(Density),:] #430

# Make Box Centered at the Point of Interest or High Density Region

# all positions are relative to the 'Center'
VoronoiPos-=Center
Pos-=Center

VoronoiPos *= 1.0 #3.086e+18                  # in cgs
Pos        *= 1.0 #3.086e+18                  # in cgs

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

print("Cores Available: ", os.cpu_count())
print("Cores Used: ", os.cpu_count())
print("Steps in Simulation: ", N)
print("rloc_boundary      : ", rloc_boundary)
print("rloc_center        : ", rloc_center)
print("max_cycles         : ", max_cycles)
print("\nBoxsize: ", Boxsize) # 256
print("Center: ", Center) # 256
print("Position of Max Density: ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume: ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume: ", Volume[np.argmax(Volume)],"\n") # 256

def get_along_lines(x_init):

    m = x_init.shape[0] # = 1

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))

    line[0,:,:]     =x_init
    line_rev[0,:,:] =x_init

    x = x_init

    print(x_init[0,:])

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos)
    dummy, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos)

    # propagates from same inner region to the outside in -dx direction
    
    for k in range(N):

        dx = 1.0    
        x, bfield, dens = Heun_step(x, dx, Bfield, Density, Density_grad, VoronoiPos)
        
        line[k+1,:,:] = x
        bfields[k+1,:] = bfield
        densities[k+1,:] = dens
    # propagates from same inner region to the outside in -dx direction
    
    for k in range(N):
        dx = -1.0
        x, bfield, dens = Heun_step(x, dx, Bfield, Density, Density_grad, VoronoiPos)
        
        line_rev[k+1,:,:] = x
        bfields_rev[k+1,:] = bfield
        densities_rev[k+1,:] = dens
        

    line_rev = line_rev[1:,:,:]
    bfields_rev = bfields_rev[1:,:] 
    densities_rev = densities_rev[1:,:]

    dens_min = np.log10(min(np.min(densities),np.min(densities_rev)))
    dens_max = np.log10(max(np.max(densities),np.max(densities_rev)))

    dens_diff = dens_max - dens_min

    path           = np.append(line, line_rev, axis=0)
    path_bfields   = np.append(bfields, bfields_rev, axis=0)
    path_densities = np.append(densities, densities_rev, axis=0)

    for j, _ in enumerate(path[0,:,0]):
        # for trajectory 
        radius_vector      = np.zeros_like(path[:,j,:])
        magnetic_fields    = np.zeros_like(path_bfields[:,j])
        gas_densities      = np.zeros_like(path_densities[:,j])
        trajectory         = np.zeros_like(path[:,j,0])

        prev_radius_vector = path[0,j,:]
        diff_rj_ri = 0.0

        for k, pk in enumerate(path[:,j,0]):
            
            radius_vector[k]    = path[k,j,:]
            magnetic_fields[k]  = path_bfields[k,j]
            gas_densities[k]    = path_densities[k,j]
            diff_rj_ri = magnitude(radius_vector[k], prev_radius_vector)
            trajectory[k] = trajectory[k-1] + diff_rj_ri
            #print(radius_vector[k], magnetic_fields[k], gas_densities[k], diff_rj_ri)
            
            prev_radius_vector  = radius_vector[k] 

        trajectory[0] *= 0.0
    
    index = len(line_rev[:,0,0])

    return index, line[0,0,:], bfields[0,0], radius_vector, trajectory, magnetic_fields, gas_densities

if sys.argv[-1] == "-1":
    # Assuming your JSON file is named 'data.json'
    file_path = 'random_distributed_reduction_factor.json'

    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Load the JSON data into a Python list
        reduction_factor = np.array(json.load(file))

    max_cycles = 0

# Generate a list of tasks
tasks = []
for i in range(max_cycles):
    
    rloc_center      = float(random.uniform(0,rloc_center))
    
    if True:
        nside = 8     # sets number of cells sampling the spherical boundary layers = 12*nside**2
        npix  = 12 * nside ** 2 
        ipix_center       = np.arange(npix)
        xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix_center)
        xx = np.array(random.sample(sorted(xx),1))
        yy = np.array(random.sample(sorted(yy),1))
        zz = np.array(random.sample(sorted(zz),1))

    m = len(zz) # amount of values that hold which_up_down

    x_init = np.zeros((m,3))
    x_init[:,0]      = rloc_center * xx[:]
    x_init[:,1]      = rloc_center * yy[:]
    x_init[:,2]      = rloc_center * zz[:]

    initial_conditions = (x_init)
    print("rloc_center:= ", rloc_center, list(x_init[0]))
    tasks.append((initial_conditions))
    
# Number of worker processes
import os
num_workers = 6#int(os.cpu_count())

# Record the start time
start_time = time.time()

# Create a Pool of worker processes
with Pool(num_workers) as pool:
    # Map the euler_integration function to the list of tasks
    results = pool.map(get_along_lines, tasks)

# Record the end time
elapsed_time = time.time() - start_time

# Print elapsed time
print(f"Elapsed time: {elapsed_time/60} Minutes")


for i, pack_dist_field_dens in enumerate(results):
    
    lmn, x_init, B_init, radius_vector, distance, bfield, numb_density = pack_dist_field_dens

    """# Obtained position along the field lines, now we find the pocket"""

    #index_peaks, global_info = pocket_finder(bfield) # this plots
    pocket, global_info = pocket_finder(bfield, cycle, plot=True) # this plots
    index_pocket, field_pocket = pocket[0], pocket[1]

    globalmax_index = global_info[0]
    globalmax_field = global_info[1]

    # Calculate the range within the 80th percentile
    start_index = len(bfield) // 10  # Skip the first 10% of indices
    end_index = len(bfield) - start_index  # Skip the last 10% of indices

    # we gotta find peaks in the interval   (B_l < random_element < B_h)
    # Generate a random index within the range
    B_r = B_init.copy()

    print("random index: ", lmn, "peak's index: ", index_pocket, field_pocket)

    """How to find index of Bl?"""

    # Bl it is definitely between two peaks, we need to verify is also inside a pocket
    # such that Bl < Bs < Bh (p_i < lmn < p_j)

    if len(index_pocket) > 1: # if there is more than 2 peaks then this is obtainable
        p_i = find_insertion_point(index_pocket, lmn)
        closest_values = index_pocket[max(0, p_i): min(len(index_pocket), p_i +2)]
        print("Random Index:", lmn, "assoc. B(s_r):",B_r)
        print("Maxima Values related to pockets: ",len(index_pocket), p_i)
    else:
        print("Random Index:", lmn, "assoc. B(s_r):",B_r)
        print("No Pockets, R = 1.")

        reduction_factor = np.append(reduction_factor, 1.)
        cycle += 1
        continue

    if len(closest_values) == 2:
        B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
        B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
    else:
        reduction_factor = np.append(reduction_factor, 1.)
        cycle += 1
        continue

    if B_r/B_l <= 1:
        R = 1 - np.sqrt(1-B_r/B_l)
        reduction_factor = np.append(reduction_factor, R)
        cycle += 1
        print("Bl: ", B_l, " B_r/B_l =", B_r/B_l, "< 1 ") 
    else:
        R = 1
        reduction_factor = np.append(reduction_factor, 1.)
        cycle += 1
        print("Bl: ", B_l, " B_r/B_l =", B_r/B_l, "> 1 so CRs are not affected => R = 1") 
    
    print("Closest local maxima 'p':", closest_values)
    print("Bs: ", bfield[lmn], "Bi: ", bfield[closest_values[0]], "Bj: ", bfield[closest_values[1]])
    
    """
    bs: where bs is the field magnitude at the random point chosen 
    bl: magnetic at position s of the trajectory
    """
    
    if True:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        axs[0].plot(distance, bfield, linestyle="--", color="m")
        axs[0].scatter(distance, bfield, marker="+", color="m")
        axs[0].set_xlabel("trajectory (pc)")
        axs[0].set_ylabel("$B(s)$ (cgs units )")
        axs[0].set_title("Individual Magnetic Field Shape")
        #axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(distance, numb_density, linestyle="--", color="m")
        axs[1].set_xlabel("trajectory (cgs units Au)")
        axs[1].set_ylabel("$n_g(s)$ Field (cgs units $M_{sun}/pc^3$) ")
        axs[1].set_title("Gas Density along Magnetic Lines")
        #axs[1].legend()
        axs[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig("field_shapes/field-density_shape.png")

        # Show the plot
        #plt.show()
        plt.close(fig)
    
    if False:
        ax = plt.figure().add_subplot(projection='3d')

        for k in range(1):
            print(k)
            x=distance[:,k,0] # 1D array
            y=distance[:,k,1]
            z=distance[:,k,2]
            
            which = x**2 + y**2 + z**2 <= rloc_boundary**2
            
            x=x[which]
            y=y[which]
            z=z[which]
            
            for l in range(len(z)):
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color="m",linewidth=0.3)

            #ax.scatter(x_init[0], x_init[1], x_init[2], marker="v",color="m",s=10)
            ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
            ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)

        ax.set_xlim(-rloc_boundary,rloc_boundary)
        ax.set_ylim(-rloc_boundary,rloc_boundary)
        ax.set_zlim(-rloc_boundary,rloc_boundary)
        ax.set_xlabel('x [pc]')
        ax.set_ylabel('y [pc]')
        ax.set_zlabel('z [pc]')
        ax.set_title('From Core to Outside in +s, -s directions')
        #plt.savefig(f'field_shapes/MagneticFieldThreading.png',bbox_inches='tight')
        #plt.close()
        plt.show()

print(reduction_factor)

# Specify the file path
file_path = f'random_distributed_reduction_factor{sys.argv[-1]}.json'

# Write the list data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(reduction_factor.tolist(), json_file)

"""# Graphs"""

#plot_trajectory_versus_magnitude(distance, bfield, ["B Field Density in Path", "B-Magnitude", "s-coordinate"])

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
#plt.savefig("c_output_data/histogramdata={len(reduction_factor)}bins={bins}"+name+".png")
plt.savefig(f"histograms/hist={len(reduction_factor)}bins={bins}.png")

# Show the plot
#plt.show()