from collections import defaultdict
from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
from scipy import spatial
import healpy as hp
import numpy as np
import random
import os
import h5py
import json
import sys
from library import *

import time

start_time = time.time()


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
	max_cycles   =int(sys.argv[3])
else:
    N            =100
    rloc_boundary=256   # rloc_boundary for boundary region of the cloud
    max_cycles   =1

reduction_factor_at_gas_density = defaultdict()

reduction_factor = np.array([])

"""
(Original Functions Made By A. Mayer (Max Planck Institute) + contributions B. E. Velazquez (University of Texas))
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
	dist, cells = spatial.KDTree(Pos[:]).query(x, k=1,workers=-1)
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
		dx = 0.4*((4/3)*Volume[cells][0]/np.pi)**(1/3)
	else:
		dx = -0.4*((4/3)*Volume[cells][0]/np.pi)**(1/3)

	x_tilde = x + dx * local_fields_1
	local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos)
	local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
	abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1))

	unitary_local_field = (local_fields_1 + local_fields_2) / np.tile(abs_sum_local_fields, (3,1)).T
	x_final = x + 0.5 * dx * unitary_local_field[0]
	
	x_final = x + 0.5 * dx * (local_fields_1 + local_fields_2)
	
	return x_final, abs_local_fields_1, local_densities

"""  B. Jesus Velazquez """

snap = '430'
filename = 'arepo_data/snap_'+ snap + '.hdf5'

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)

# Initialize gradients
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))

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


grams to nucleus/cm^3
"""

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

Bfield  *= 1.0 * (3.086e+18/1.9885e33)**(-1/2) # in cgs
Density *= 1.0 * 6.771194847794873e-23
Mass    *= 1.0 * 1.9885e33
Volume   = Mass/Density

#Center= 0.5 * Boxsize * np.ones(3) # Center
#Center = np.array( [91,       -110,          -64.5]) #117
#Center = np.array( [96.6062303,140.98704002, 195.78020632]) #117
Center = Pos[np.argmax(Density),:] #430
print("Center before Centering", Center)
# Make Box Centered at the Point of Interest or High Density Region

# all positions are relative to the 'Center'
VoronoiPos-=Center
Pos-=Center

VoronoiPos *= 1.0*1.496e13
Pos        *= 1.0*1.496e13

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

def get_along_lines(x_init):

    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))

    line[0,:,:]     =x_init
    line_rev[0,:,:] =x_init

    x = x_init

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Density_grad, Pos)

    # propagates from same inner region to the outside in -dx direction
    start_time = time.time()
    for k in range(N):
        #print(k, (time.time()-start_time)/60.)
        
        x, bfield, dens = Heun_step(x, 1, Bfield, Density, Density_grad, VoronoiPos)
        print(k, x, bfield, dens)
        line[k+1,:,:] = x
        bfields[k+1,:] = bfield
        densities[k+1,:] = dens

    # propagates from same inner region to the outside in -dx direction
    x = x_init

    dummy, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Density_grad, Pos)
	
    for k in range(N):
        #print(-k, (time.time()-start_time)/60.)
        x, bfield, dens = Heun_step(x, -1, Bfield, Density, Density_grad, VoronoiPos)
        print(-k, x, bfield, dens)
        line_rev[k+1,:,:] = x
        bfields_rev[k+1,:] = bfield
        densities_rev[k+1,:] = dens

    line_rev = line_rev[:,:,:]
    bfields_rev = bfields_rev[:,:] 
    densities_rev = densities_rev[:,:]
	
    # Concatenating the arrays correctly as 3D arrays
    # Concatenate the `line` and `line_rev` arrays along the first axis, but only take the first element in the `m` dimension
    radius_vector = np.concatenate((line[:, :, :], line_rev[::-1, :, :]), axis=0)
    radius_vector = line[:, :, :]

    # Concatenate the `bfields` and `bfields_rev` arrays along the first axis, but only take the first element in the `m` dimension
    magnetic_fields = np.concatenate((bfields[:, :], bfields_rev[::-1, :]), axis=0)
    magnetic_fields = bfields[:, :]

    # Concatenate the `densities` and `densities_rev` arrays along the first axis, but only take the first element in the `m` dimension
    gas_densities = np.concatenate((densities[:, :], densities_rev[::-1, :]), axis=0)
    gas_densities = densities[:, :]

    trajectory         = np.zeros(bfields.shape)        # 1D array for the trajectory
        
    for _n in range(m):  # Iterate over the first dimension
        prev = radius_vector[0, _n, :]
        for k in range(N):  # Iterate over the first dimension    
            cur = radius_vector[k, _n, :]
            
            diff_rj_ri = magnitude(cur, prev)
            trajectory[k,_n] = trajectory[k-1,_n] + diff_rj_ri            
            prev = radius_vector[k, _n,:]

    #index = len(line_rev[:, 0, 0])
    trajectory[-1,:] = 2*trajectory[-2,:] - trajectory[-3,:] 
    return bfields[0,:], radius_vector, trajectory, magnetic_fields, gas_densities
    #return index, line[0, :], bfields[0], radius_vector, trajectory, magnetic_fields, gas_densities

import os
import shutil

# Define the directory names
output_folder = 'arepo_output_data'
new_folder = 'arepo_npys'

# Check if the arepo_output_data folder exists
if os.path.exists(output_folder):
    # Delete the folder and its contents
    shutil.rmtree(output_folder)

# Create the new arepo_npys directory
os.makedirs(new_folder, exist_ok=True)
rloc_center      = np.array([float(random.uniform(0,rloc_boundary)) for l in range(max_cycles)])
nside = max_cycles     # sets number of cells sampling the spherical boundary layers = 12*nside**2
npix  = 12 * nside ** 2
ipix_center       = np.arange(npix)
xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix_center)

xx = np.array(random.sample(list(xx), max_cycles))
yy = np.array(random.sample(list(yy), max_cycles))
zz = np.array(random.sample(list(zz), max_cycles))

m = len(zz) # amount of values that hold which_up_down

x_init = np.zeros((m,3))

x_init[:,0]      = rloc_center * xx[:]
x_init[:,1]      = rloc_center * yy[:]
x_init[:,2]      = rloc_center * zz[:]

lmn = N - 1

print("Cores Used: ", os.cpu_count())
print("Steps in Simulation: ", 2*N)
print("rloc_boundary      : ", rloc_boundary)
print("rloc_center        : ", rloc_center)
print("max_cycles         : ", max_cycles)
print("Boxsize            : ", Boxsize) # 256
print("Center             : ", Center) # 256
print("Posit Max Density  : ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume    : ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume    : ", Volume[np.argmax(Volume)]) # 256
print(f"Smallest Density   : {Density[np.argmin(Density)]}")
print(f"Biggest  Density   : {Density[np.argmax(Density)]}")

B_init, radius_vector, trajectory, magnetic_fields, gas_densities = get_along_lines(x_init)

print("Elapsed Time: ", (time.time() - start_time)/60.)

with open('output', 'w') as file:
    file.write(f"{filename}\n")
    file.write(f"Cores Used: {os.cpu_count()}\n")
    file.write(f"Steps in Simulation: {2 * N}\n")
    file.write(f"rloc_boundary      : {rloc_boundary}\n")
    file.write(f"rloc_center        : {rloc_center}\n")
    file.write(f"max_cycles         : {max_cycles}\n")
    file.write(f"Boxsize            : {Boxsize}\n")
    file.write(f"Center             : {Center}\n")
    file.write(f"Posit Max Density  : {Pos[np.argmax(Density), :]}\n")
    file.write(f"Smallest Volume    : {Volume[np.argmin(Volume)]}\n")
    file.write(f"Biggest  Volume    : {Volume[np.argmax(Volume)]}\n")
    file.write(f"Smallest Density   : {Density[np.argmin(Density)]}\n")
    file.write(f"Biggest  Density   : {Density[np.argmax(Density)]}\n")
    file.write(f"Elapsed Time       : {(time.time() - start_time)/60.}\n")

for i in range(m):

    pocket, global_info = pocket_finder(magnetic_fields[:,i], i, plot=True) # this plots

    np.save(f"arepo_npys/ArePositions{i}.npy", radius_vector[:,i,:])
    np.save(f"arepo_npys/ArepoTrajectory{i}.npy", trajectory[:,i])
    np.save(f"arepo_npys/ArepoNumberDensities{i}.npy", gas_densities[:,i])
    np.save(f"arepo_npys/ArepoMagneticFields{i}.npy", magnetic_fields[:,i])
	
    print(f"finished line {i+1}/{max_cycles}",(time.time()-start_time)/60)

    if True:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        axs[0].plot(trajectory[:,i], magnetic_fields[:,i], linestyle="--", color="m")
        axs[0].scatter(trajectory[:,i], magnetic_fields[:,i], marker="+", color="m")
        axs[0].set_xlabel("trajectory (Pc)")
        axs[0].set_ylabel("$B(s)$ (Gauss (M_{sun}/Pc)**(1/2))")
        axs[0].set_title("Individual Magnetic Field Shape")
        axs[0].grid(True)

        axs[1].plot(trajectory[:,i], gas_densities[:,i], linestyle="--", color="m")
        axs[1].scatter(trajectory[:,i], gas_densities[:,i], marker="+", color="m")
        axs[1].set_xlabel("trajectory (cgs units Pc)")
        axs[1].set_ylabel("$n_g(s)$ Field ($M_{sun}/Pc^3$) ")
        axs[1].set_title("Gas Density along Magnetic Lines")
        axs[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"field_shapes/field-density_shape{i}.png")

        # Show the plot
        #plt.show()
        plt.close(fig)


if True:
	ax = plt.figure().add_subplot(projection='3d')

	for k in range(1):
		x=trajectory[:,0] # 1D array
		y=trajectory[:,1]
		z=trajectory[:,2]
		
		#which = x**2 + y**2 + z**2 <= rloc_boundary**2
		
		#x=x[which]
		#y=y[which]
		#z=z[which]
		
		for l in range(len(z)):
			ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color="m",linewidth=0.3)

		#ax.scatter(x_init[0], x_init[1], x_init[2], marker="v",color="m",s=10)
		ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
		ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
		

	ax.set_xlim(-rloc_boundary,rloc_boundary)
	ax.set_ylim(-rloc_boundary,rloc_boundary)
	ax.set_zlim(-rloc_boundary,rloc_boundary)
	ax.set_xlabel('x [AU]')
	ax.set_ylabel('y [AU]')
	ax.set_zlabel('z [AU]')
	ax.set_title('From Core to Outside in +s, -s directions')

	plt.savefig(f'field_shapes/MagneticFieldTopology.png',bbox_inches='tight')

	#plt.close()
	#plt.show()

