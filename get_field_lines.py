import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import random
import h5py
import sys
import os
import glob
from library import *

import time

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

"""  
Analysis of reduction factor

$$N(s) 1 - \sqrt{1-B(s)/B_l}$$

"""

start_time = time.time()

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

def find_points_and_relative_positions(x, Pos, VoronoiPos):
	dist, cells = spatial.KDTree(Pos[:]).query(x, k=1,workers=-1)
	rel_pos = VoronoiPos[cells] - x
	return dist, cells, rel_pos

def find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos, VoronoiPos)
	local_fields = get_magnetic_field_at_points(x, Bfield[cells], rel_pos)
	local_densities = get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos)
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	return local_fields, abs_local_fields, local_densities, cells
	
def Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume, bdirection = None):
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T
    CellVol = Volume[cells]
    dx *= ((3/4)*Volume[cells]/np.pi)**(1/3)  
    x_tilde = x + dx[:, np.newaxis] * local_fields_1
    local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
    abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1))

    unito = 2*(local_fields_1 + local_fields_2)/abs_sum_local_fields[:, np.newaxis]
    x_final = x + 0.5 * dx[:, np.newaxis] * unito

    bdirection = 0.5*(local_fields_1 + local_fields_2)
    
    return x_final, abs_local_fields_1, local_densities, CellVol


FloatType = np.float64
IntType = np.int32

if len(sys.argv)>2:
	# first argument is a number related to rloc_boundary
	N=int(sys.argv[1])
	rloc_boundary=float(sys.argv[2])
	max_cycles   =int(sys.argv[3])
	num_file = str(sys.argv[4])
	case = str(sys.argv[5])
else:
    N            =2000
    rloc_boundary=1   # rloc_boundary for boundary region of the cloud
    max_cycles   =10
    num_file = '430'
    case = 'amb'

if case == 'ideal':
    subdirectory = 'ideal_mhd'
elif case == 'amb':
    subdirectory = 'ambipolar_diffusion'
else:
    subdirectory= ''

file_list = glob.glob(f'arepo_data/{subdirectory}/*.hdf5')
print(file_list)
filename = None

for f in file_list:
    if num_file in f:
        filename = f
if filename == None:
    raise FileNotFoundError
    
new_folder = os.path.join(f"getLines/{case}/" , num_file)
os.makedirs(new_folder, exist_ok=True)

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))
Volume   = Mass/Density

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

file_path       = f'cloud_tracker_slices/{case}/{case}_cloud_trajectory.txt'

import csv

snap = []
time_value = []

with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        if num_file == str(row[0]):
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])
            snap =str(row[0])
            time_value = float(row[1])
            peak_den =  float(row[5])

CloudCord = Center.copy()

print("Center before Centering", Center)

VoronoiPos-=CloudCord
Pos-=CloudCord

for dim in range(3):  # Loop over x, y, z
    pos_from_center = Pos[:, dim]
    boundary_mask = pos_from_center > Boxsize / 2
    Pos[boundary_mask, dim] -= Boxsize
    VoronoiPos[boundary_mask, dim] -= Boxsize

densthresh = 100

rloc_boundary = 1.0

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

        line[k+1,:,:]    = x
        volumes[k+1,:]   = vol
        bfields[k+1,:]   = bfield
        densities[k+1,:] = dens

        step_diff = max_threshold-unique_unmasked_max_threshold
        
        order_clause = step_diff >= 1_000
        percentage_clause = np.sum(un_masked)/len(mask) > 0.8

        if np.all(un_masked) or (order_clause and percentage_clause): 
            if (order_clause and percentage_clause):
                with open(f'isolated_radius_vectors{num_file}.dat', 'a') as file: 
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
                with open(f'isolated_radius_vectors{num_file}.dat', 'a') as file: 
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
    
    radius_vector = np.append(line_rev[::-1, :, :], line[1:,:,:], axis=0)
    magnetic_fields = np.append(bfields_rev[::-1, :], bfields[1:,:], axis=0)
    numb_densities = np.append(densities_rev[::-1, :], densities[1:,:], axis=0)
    volumes_all = np.append(volumes_rev[::-1, :], volumes[1:,:], axis=0)

    trajectory = np.zeros_like(magnetic_fields)
    column = np.zeros_like(magnetic_fields)
    radius_to_origin = np.zeros_like(magnetic_fields)

    print("Magnetic fields shape:", magnetic_fields.shape)
    print("Radius vector shape:", radius_vector.shape)
    print("Numb densities shape:", numb_densities.shape)

    m = magnetic_fields.shape[1]
    print("Surviving lines: ", m, "out of: ", max_cycles)

    radius_vector   *= 1.0* 3.086e+18                                # from Parsec to cm
	
    for _n in range(m):  # Iterate over the first dimension
        prev = radius_vector[0, _n, :]
        trajectory[0, _n] = 0  # Initialize first row
        column[0, _n] = 0      # Initialize first row
        
        for k in range(1, magnetic_fields.shape[0]):  # Start from k = 1 to avoid indexing errors
            radius_to_origin[k, _n] = magnitude(radius_vector[k, _n, :])
            
            cur = radius_vector[k, _n, :]
            diff_rj_ri = magnitude(cur - prev)  # Vector subtraction before calculating magnitude

            trajectory[k, _n] = trajectory[k-1, _n] + diff_rj_ri            
            column[k, _n] = column[k-1, _n] + numb_densities[_n, k] * diff_rj_ri            
            
            prev = cur  # Store current point as previous point

    volumes_all     *= 1.0#/(3.086e+18**3) 
    trajectory      *= 1.0#* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0#* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)

    return radius_vector, trajectory, magnetic_fields, numb_densities, volumes_all, radius_to_origin, [threshold, threshold_rev], column

def generate_vectors_in_core(max_cycles, densthresh, rloc=1.0, seed=12345):
    import numpy as np
    from scipy.spatial import cKDTree
    np.random.seed(seed)
    valid_vectors = []
    tree = cKDTree(Pos)
    while len(valid_vectors) < max_cycles:
        points = np.random.uniform(low=-rloc, high=rloc, size=(max_cycles, 3))
        distances = np.linalg.norm(points, axis=1)
        inside_sphere = points[distances <= rloc]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > densthresh
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
    valid_vectors = np.array(valid_vectors)
    random_indices = np.random.choice(len(valid_vectors), max_cycles, replace=False)
    return valid_vectors[random_indices]

x_init = generate_vectors_in_core(max_cycles, densthresh)

print("Cores Used         : ", os.cpu_count())
print("Steps in Simulation: ", 2*N)
print("rloc_boundary      : ", rloc_boundary)
print("x_init             : ", x_init)
print("max_cycles         : ", max_cycles)
print("Boxsize            : ", Boxsize) # 256
print("Center             : ", CloudCord) # 256
print("Posit Max Density  : ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume    : ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume    : ", Volume[np.argmax(Volume)]) # 256
print(f"Smallest Density  : {Density[np.argmin(Density)]}")
print(f"Biggest  Density  : {Density[np.argmax(Density)]}")

radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th, cd  = get_along_lines(x_init)

m = magnetic_fields.shape[1]

threshold, threshold_rev = th

print("Elapsed Time: ", (time.time() - start_time)/60.)

for i in range(m):

    print(f"{threshold[i]} - {threshold_rev[i]}")
    _from = N+1 - threshold_rev[i]
    _to   = N+1 + threshold[i]
    print(f"{_from} - {_to}")

    column_dens  = cd[_from:_to,i]
    mag_field    = magnetic_fields[_from:_to,i]
    pos_vector   = radius_vector[_from:_to,i,:] # stays in Pc
    path_distance = trajectory[_from:_to,i] - trajectory[_from,i] # stays in Pc
    mass_density = numb_densities[_from:_to,i] /gr_cm3_to_nuclei_cm3
    numb_density = numb_densities[_from:_to,i] 
    volume       = volumes[_from:_to,i]*(parsec_to_cm3**3)

    if len(trajectory) == 0:
        continue

    print("column_dens shape:", column_dens.shape)
    print("mag_field shape:", mag_field.shape)
    print("pos_vector shape:", pos_vector.shape)
    print("path_distance shape:", path_distance.shape)
    print("numb_density shape:", numb_density.shape)
    print("volume shape:", volume.shape)

    pocket, global_info = pocket_finder(mag_field, i, plot=False) # this plots

    np.save(os.path.join(new_folder, f"ColumnDensity{i}.npy"), column_dens)
    np.save(os.path.join(new_folder, f"Positions{i}.npy"), pos_vector)
    np.save(os.path.join(new_folder, f"Trajectory{i}.npy"), path_distance)
    np.save(os.path.join(new_folder, f"NumberDensities{i}.npy"), numb_density)
    np.save(os.path.join(new_folder, f"MagneticFields{i}.npy"), mag_field)
    np.save(os.path.join(new_folder, f"Volumes{i}.npy"), volume)

	
    print(f"finished line {i+1}/{max_cycles}",(time.time()-start_time)/60)

    if True:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        axs[0,0].plot(path_distance*length_unit, mag_field, linestyle="--", color="m")
        axs[0,0].set_xlabel("s (cm)")
        axs[0,0].set_ylabel("$B(s)$ $\mu$ G (cgs)")
        axs[0,0].set_title("Magnetic FIeld")
        axs[0,0].grid(True)
		
        axs[0,1].plot(path_distance*length_unit, linestyle="--", color="m")
        axs[0,1].set_xlabel("# steps")
        axs[0,1].set_ylabel("$s$ cm")
        axs[0,1].set_title("Distance Away of $n_g^{max}(r)$ ")
        axs[0,1].grid(True)

        axs[1,0].plot(path_distance*length_unit, numb_density, linestyle="--", color="m")
        axs[1,0].set_yscale('log')
        axs[1,0].set_xlabel("s (cm)")
        axs[1,0].set_ylabel("$N_g(s)$ cm^-3")
        axs[1,0].set_title("Number Density (Nucleons/cm^3) ")
        axs[1,0].grid(True)
		
        axs[1,1].plot(volume, linestyle="-", color="black")
        axs[1,1].set_yscale('log')
        axs[1,1].set_xlabel("# steps")
        axs[1,1].set_ylabel("$V(s) cm^3 $ (cgs)")
        axs[1,1].set_title("Cells Volume along Path")
        axs[1,1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        plt.savefig(os.path.join(new_folder,f"mosaic{i}.png"))
        
        # Close the plot
        plt.close(fig)
    
    if True:
            
        from matplotlib import cm
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=np.min(magnetic_fields), vmax=np.max(magnetic_fields))
        cmap = cm.viridis

        ax = plt.figure().add_subplot(projection='3d')
        radius_vector /= 3.086e+18

        for k in range(m):
            x=radius_vector[:, k, 0]
            y=radius_vector[:, k, 1]
            z=radius_vector[:, k, 2]
            
            for l in range(len(x)):
                color = cmap(norm(magnetic_fields[l, k]))
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color,linewidth=0.3)

            #ax.scatter(x_init[0], x_init[1], x_init[2], marker="v",color="m",s=10)
            ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
            ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
                
        radius_to_origin = np.sqrt(x**2 + y**2 + z**2)
        zoom = np.max(radius_to_origin)
        ax.set_xlim(-zoom,zoom)
        ax.set_ylim(-zoom,zoom)
        ax.set_zlim(-zoom,zoom)
        ax.set_xlabel('x [Pc]')
        ax.set_ylabel('y [Pc]')
        ax.set_zlabel('z [Pc]')
        ax.set_title('Magnetic field morphology')

        # Add a colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Magnetic Field Strength')
        plt.savefig(os.path.join(new_folder,f"FieldTopology{i}.png"), bbox_inches='tight')
        plt.show()