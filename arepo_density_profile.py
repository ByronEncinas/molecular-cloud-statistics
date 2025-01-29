import matplotlib.pyplot as plt
from scipy import spatial
import healpy as hp
import numpy as np
import h5py
import sys
import os

from library import *

import time

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
	
def Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume):
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T
    CellVol = Volume[cells]
    dx *= 0.4*((3/4)*CellVol/np.pi)**(1/3)  
    x_tilde = x + dx[:, np.newaxis] * local_fields_1
    local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos, VoronoiPos)
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
    abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1))

    unito = 2*(local_fields_1 + local_fields_2)/abs_sum_local_fields[:, np.newaxis]
    x_final = x + 0.5 * dx[:, np.newaxis] * unito
    kinetic_energy = 0.5*Mass[cells]*np.linalg.norm(Velocities[cells], axis=1)**2
    internal_energy = InternalEnergy[cells]
    pressure = Pressure[cells]
    
    return x_final, abs_local_fields_1, local_densities, CellVol, kinetic_energy, internal_energy, pressure

"""  
Using Alex Mayer Data

"""

"""
Parameters

- [N] default total number of steps in the simulation
- [dx] default 4/N of the rloc_boundary (radius of spherical region of interest) variable

"""
FloatType = np.float64
IntType = np.int32

if len(sys.argv)>2:
    N             = int(sys.argv[1])
    typpe         = str(sys.argv[2]) #ideal/amb
    num_file          = str(sys.argv[3]) 
else:
    N            = 4_000
    typpe        = 'ideal'
    num_file     = '430'

"""  B. Jesus Velazquez    """

if typpe == 'ideal':
    subdirectory = 'ideal_mhd'
elif typpe == 'amb':
    subdirectory = 'ambipolar_diffusion'
else:
    subdirectory= ''

trajectory_path = f'cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt'

import csv
import numpy as np

# Path to the input file
file_path = f'cloud_tracker_slices/{typpe}/{typpe}_cloud_trajectory.txt'

# Lists to store column data
snap = []
time_value = []

# Open the file and read it using the CSV module
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
    next(csv_reader)  # Skip the header row
    
    # Read each row of data
    for row in csv_reader:
        snap.append(int(row[0]))  # First column is snap
        time_value.append(float(row[1]))  # Second column is time_value
        if num_file == str(row[0]):
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])

print(Center)

# Convert lists to numpy arrays
snap_array = np.array(snap)
time_value_array = np.array(time_value)

import glob

# Get the list of files from the directory
directory_path = f"arepo_data/{subdirectory}"
file_list = glob.glob(f"{directory_path}/*.hdf5")

# Print the first 5 files for debugging/inspection
print(file_list[:5])

filename = None

for f in file_list:
    if num_file in f:
        filename = f

snap = filename.split(".")[0][-3:]

# Create the directory path
output_path = os.path.join(f"./density_profiles/{typpe}/", snap)

# Check if the directory exists, if not create it
if not os.path.exists(output_path):
    os.makedirs(directory_path, exist_ok=True)
    print(f"Directory {output_path} created.")
else:
    print(f"Directory {output_path} already exists.")

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Velocities = np.asarray(data['PartType0']['Velocities'], dtype=FloatType)
InternalEnergy = np.asarray(data['PartType0']['InternalEnergy'], dtype=FloatType)
Pressure = np.asarray(data['PartType0']['Pressure'], dtype=FloatType)
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

Volume   = Mass/Density

print("Center before Centering", Center)

VoronoiPos-=Center
Pos-=Center

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]        -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

# Original direction vectors
axis = np.array([[ 1.0,  0.0,  0.0], # +x 0
                [ 0.0,  1.0,  0.0],  # +y 1
                [ 0.0,  0.0,  1.0],  # +z 2
                [-1.0,  0.0,  0.0],  # -x 3
                [ 0.0, -1.0,  0.0],  # -y 4
                [ 0.0,  0.0, -1.0]]) # -z 5

# Diagonal vectors
diagonals = np.array([[ 1.0,  1.0,  1.0],
                    [ 1.0,  1.0, -1.0],
                    [ 1.0, -1.0,  1.0],
                    [ 1.0, -1.0, -1.0],
                    [-1.0,  1.0,  1.0],
                    [-1.0,  1.0, -1.0],
                    [-1.0, -1.0,  1.0],
                    [-1.0, -1.0, -1.0]])

# Normalize the diagonal vectors to make them unit vectors
unit_diagonals = diagonals / np.linalg.norm(diagonals[0])

# Combine both arrays
directions= np.vstack((axis, unit_diagonals))

directions = fibonacci_sphere(14)

m = directions.shape[0]

x_init = np.zeros((m,3))

print(directions.shape)

def store_in_directory():
    import os
    import shutil

    new_folder = os.join("density_profiles/" , snap)
    # Create the new arepo_npys directory
    os.makedirs(new_folder, exist_ok=True)

def get_along_lines(x_init):
    
    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))
    volumes   = np.zeros((N+1,m))
    threshold = np.zeros((m,)).astype(int) # one value for each

    energy_magnetic   = np.zeros((N+1,m))
    energy_grav   = np.zeros((N+1,m))
    energy_thermal   = np.zeros((N+1,m))
    eff_column_densities   = np.zeros((N+1,m))
    
    line[0,:,:]     = x_init
    x = x_init
    dummy, bfields[0,:], dens, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)
    vol = Volume[cells]
    mass_dens = dens * code_units_to_gr_cm3
    dens = dens* gr_cm3_to_nuclei_cm3
    densities[0,:] = dens
    dx_vec = 0.3 * ((4 / 3) * vol / np.pi) ** (1 / 3)

    rad = np.linalg.norm(x, axis=1)
    pressure = Pressure[cells]* mass_unit / (length_unit * (time_unit ** 2))  #cgs
    m_H = 1.0079 # for atomic hydrogen
    mu = 1.0
    grav_potential = 0.0

    energy_magnetic[0,:] = bfields[0,:]*bfields[0,:]/(8*np.pi)*(vol*parsec_to_cm3**3)
    energy_thermal[0,:]  = (3 / 2) * pressure * (4*np.pi*(rad*parsec_to_cm3)**2*(dx_vec*parsec_to_cm3))
    energy_grav[0,:]     = 0.0
    
    k=0

    mask = dens > 100 # True if continue
    un_masked = np.logical_not(mask)

    while np.any((mask)):
        # Update mask for values that are 10^2 N/cm^3 above the threshold
        mask = dens > 100  # True if continue
        un_masked = np.logical_not(mask)

        # Perform Heun step and update values
        _, bfield, dens, vol, ke, ie, pressure = Heun_step(
            x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
        )
        
        mass_dens = dens * code_units_to_gr_cm3
        pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens *= gr_cm3_to_nuclei_cm3
        
        vol[un_masked] = 0

        non_zero = vol > 0
        if len(vol[non_zero]) == 0:
            break

        #dx_vec = np.min(((4 / 3) * vol[non_zero] / np.pi) ** (1 / 3))  # Increment step size
        dx_vec = np.min(((4 / 3) * vol[non_zero] / np.pi) ** (1 / 3))  # Increment step size

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        x += dx_vec * directions

        line[k + 1, :, :] = x
        volumes[k + 1, :] = vol
        bfields[k + 1, :] = bfield
        densities[k + 1, :] = dens

        # Compute radial positions
        rad = np.linalg.norm(x[:, :], axis=1)

        # Gravitational potential contribution
        grav_potential += -(4 * np.pi) ** 2 * (dens ** 2) * (rad * parsec_to_cm3) ** 4 * (dx_vec * parsec_to_cm3)

        # Compute cumulative mass for gravitational potential
        M_r = np.cumsum(4 * np.pi * mass_dens * (rad* parsec_to_cm3) ** 2 * dx_vec * parsec_to_cm3)

        # Gravitational potential: phi(r)
        #phi = (grav_constant_cgs * M_r[-1] / rad[-1]) - np.cumsum(grav_constant_cgs * M_r / rad ** 2)
        #phi = (grav_constant_cgs * M_r[-1] / rad[-1]) - np.cumsum(grav_constant_cgs * M_r / rad ** 2)
        # Gravitational energy
        #grav_energy_density = 4 * np.pi * rad ** 2 * dens * phi
        # Gravitational binding energy
        
        binding_energy = -np.sum((grav_constant_cgs * M_r / (rad*parsec_to_cm3)) * 4 * np.pi * (rad*parsec_to_cm3)**2 * mass_dens * dx_vec*parsec_to_cm3)

        energy_grav[k + 1, :]     = binding_energy
        energy_magnetic[k + 1, :] = energy_magnetic[k, :] +  bfield * bfield / (8 * np.pi) * (4*np.pi*(rad*parsec_to_cm3)**2*(dx_vec*parsec_to_cm3))
        energy_thermal[k + 1, :]  = energy_thermal[k, :] + (3 / 2) * pressure * (4*np.pi*(rad*parsec_to_cm3))**2*(dx_vec*parsec_to_cm3)

        eff_column_densities[k + 1, :] = eff_column_densities[k, :] + dens * (dx_vec*parsec_to_cm3)

        print(f"Eff. Column Densities: {eff_column_densities[k + 1, 0]:5e}")

        if np.all(un_masked):
            print("All values are False: means all density < 10^2")
            break

        k += 1

    #ratio_thermal = energy_thermal / energy_grav
    #ratio_magnetic = energy_magnetic / energy_grav

    threshold = threshold.astype(int)
    larger_cut = np.max(threshold)
    
    # cut all of them to standard
    radius_vector    = line[:larger_cut+1,:,:]
    magnetic_fields  = bfields[:larger_cut+1,:]
    numb_densities   = densities[:larger_cut+1,:]
    volumes          = volumes[:larger_cut+1,:]

    # Initialize trajectory and radius_to_origin with the same shape
    trajectory      = np.zeros_like(magnetic_fields)
    radius_to_origin= np.zeros_like(magnetic_fields)

    print("Magnetic fields shape:", magnetic_fields.shape)
    print("Radius vector shape:", radius_vector.shape)
    print("Numb densities shape:", numb_densities.shape)

    for _n in range(m):  # Iterate over the second dimensio
        prev = radius_vector[0, _n, :]
        for k in range(magnetic_fields.shape[0]):  # Iterate over the first dimension
            radius_to_origin[k, _n] = np.linalg.norm(radius_vector[k, _n, :])
            cur = radius_vector[k, _n, :]
            diff_rj_ri = np.linalg.norm(cur - prev)  # Calculate the difference between consecutive points
            if k == 0:
                trajectory[k, _n] = 0.0  # Ensure the starting point of trajectory is zero
            else:
                trajectory[k, _n] = trajectory[k-1, _n] + diff_rj_ri*parsec_to_cm3
            prev = cur  # Update `prev` to the current point
            print("Trajectory at step", k, ":", trajectory[k, _n])

    return radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, threshold, [eff_column_densities, energy_magnetic, energy_thermal, energy_grav]

print("Steps in Simulation: ", N)
print("Boxsize            : ", Boxsize)
print("Smallest Volume    : ", Volume[np.argmin(Volume)])
print("Biggest  Volume    : ", Volume[np.argmax(Volume)])
print(f"Smallest Density  : {Density[np.argmin(Density)]}")
print(f"Biggest  Density  : {Density[np.argmax(Density)]}")

radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, threshold, col_energies = get_along_lines(x_init)

print("Elapsed Time: ", (time.time() - start_time)/60.)

import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)
if True:
    for i in range(m):
        eff_column_densities, energy_magnetic, energy_thermal, energy_grav = col_energies
        cut = threshold[i]
        mean_column_over_radius = np.mean(eff_column_densities[:cut-1,i], axis=1)
        print(mean_column_over_radius[-1])

        # Define mosaic layout
        mosaic = [
            ['A', 'B'],
            ['E', '.'],
            ['C', 'D']
        ]
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(12, 10), dpi=300)

        # Plot Magnetic Energy
        axs['A'].plot(trajectory[:cut, i], numb_densities[:cut, i], linestyle="--", color="blue")
        axs['A'].scatter(trajectory[:cut, i], numb_densities[:cut, i], marker="o", color="blue", s=5)
        axs['A'].set_yscale('log')
        axs['A'].set_xlabel("s (cm) along LOS")
        axs['A'].set_ylabel("$n_g(s)$")
        axs['A'].set_title("Number Density (LOS)")
        axs['A'].grid(True)

        # Plot Thermal Energy
        axs['B'].plot(trajectory[:cut, i], energy_magnetic[:cut, i], linestyle="--", color="red")
        axs['B'].set_xlabel("s (cm) along LOS")
        axs['B'].set_ylabel("Energy (ergs)")
        axs['B'].set_title("Magnetic Energy (LOS)")
        axs['B'].grid(True)

        # Energies Relative to Gravitational Energy
        axs['C'].plot(trajectory[:cut, i], energy_thermal[:cut, i], linestyle="--", color="red", label="Thermal Energy")
        axs['C'].set_xlabel("s (cm) along LOS")
        axs['C'].set_ylabel("Energy")
        axs['C'].set_title("Thermal Energy (LOS)")
        axs['C'].legend()
        axs['C'].grid(True)

        # Gravitational Energy
        axs['D'].plot(trajectory[:cut, i], energy_grav[:cut, i], linestyle="--", color="orange", label="Gravitational Energy")
        axs['D'].set_xlabel("s (cm) along LOS")
        axs['D'].set_ylabel("$E_{grav}$")
        axs['D'].set_title("Gravitational Binding Energy (LOS)")
        axs['D'].legend()
        axs['D'].grid(True)

        # Table Data
        table_data = [
            ['---', 'Value', 'Note'],
            ['Mean Column Density (LOS)', f'{mean_column_over_radius[-1]:.5e}', '-'],
            ['Magnetic Energy', f'{energy_magnetic[-1,i]:.5e}', '-'],
            ['Thermal Energy', f'{energy_thermal[-1,i]:.5e}', '-'],
            ['Grav Binding Energy', f'{energy_grav[-1,i]:.5e}', '-'],
            ['Steps in Simulation (LOS)', str(len(trajectory)), '-'],
            ['Smallest Volume (LOS)', f'{np.max(volumes[:cut,i]):.3e}', '-'],
            ['Biggest Volume (LOS)', f'{np.max(volumes[:cut,i]):.3e}', '-'],
            ['Smallest Density (LOS)', f'{np.min(numb_densities[:cut,i]):.3e}', '-'],
            ['Biggest Density (LOS)', f'{np.max(numb_densities[:cut,i]):.3e}', '-']
        ]
        table = axs['E'].table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
        axs['E'].axis('off')

        # Adjust Layout and Save Figure
        plt.tight_layout()
        plt.savefig(f"{output_path}/energies_mosaic_{i}.png", dpi=300)
        plt.close(fig)
        
        # Save column density values to a text file after plotting
        with open("column_density_values.txt", "a") as file:  # Open in append mode to avoid overwriting
            file.write(f"{mean_column_over_radius[-1]:.3e}\n")

if True:
    for i in range(m):
        mean_column, energy_magnetic, energy_thermal, energy_grav = col_energies
        cut = threshold[i]
        
        print(mean_column[-1])
        mean_grav = np.mean(energy_grav[:cut,i])
        
        # Define mosaic layout
        mosaic = [
            ['A', 'B'],
            ['C', 'D']
        ]
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(12, 10), dpi=300)

        # Plot Magnetic Energy
        axs['A'].plot(trajectory[:cut, i], numb_densities[:cut, i], linestyle="--", color="blue")
        axs['A'].scatter(trajectory[:cut, i], numb_densities[:cut, i], marker="o", color="blue", s=5)
        axs['A'].set_yscale('log')
        axs['A'].set_xlabel("s (cm) along LOS")
        axs['A'].set_ylabel("$n_g(s)$")
        axs['A'].set_title("Number Density (LOS)")
        axs['A'].grid(True)

        # Plot Thermal Energy
        axs['B'].plot(trajectory[:cut, i], energy_magnetic[:cut, i]/mean_grav, linestyle="--", color="red")
        axs['B'].set_xlabel("s (cm) along LOS")
        axs['B'].set_ylabel("ratio")
        axs['B'].set_title("$E_{mag}/\hbar{E_grav}$ Energy Ratio (LOS)")
        axs['B'].grid(True)

        # Energies Relative to Gravitational Energy
        axs['C'].plot(trajectory[:cut, i], energy_thermal[:cut, i]/mean_grav, linestyle="--", color="red", label="Thermal Energy")
        axs['C'].set_xlabel("s (cm) along LOS")
        axs['C'].set_ylabel("ratio")
        axs['C'].set_title("$E_{thermal}/\hbar{E_grav}$ Energy Ratio (LOS)")
        axs['C'].legend()
        axs['C'].grid(True)

        # Gravitational Energy
        axs['D'].plot(trajectory[:cut, i], energy_grav[:cut, i]/mean_grav, linestyle="--", color="orange", label="Gravitational Energy")
        axs['D'].set_xlabel("s (cm) along LOS")
        axs['D'].set_ylabel("$E_{grav}/\hbar{E_grav}$")
        axs['D'].set_title("Scaled Gravitational Binding Energy (LOS)")
        axs['D'].legend()
        axs['D'].grid(True)

        # Adjust Layout and Save Figure
        plt.tight_layout()
        plt.savefig(f"{output_path}/ratios_mosaic_{i}.png", dpi=300)
        plt.close(fig)

    if True:
        ax = plt.figure().add_subplot(projection='3d')
        dens_min = np.log10(np.min(numb_densities))
        dens_max = np.log10(np.max(numb_densities))

        dens_diff = dens_max - dens_min

        for k in range(m):
            x=radius_vector[:,k,0]/ 3.086e+18                                # from Parsec to cm
            y=radius_vector[:,k,1]/ 3.086e+18                                # from Parsec to cm
            z=radius_vector[:,k,2]/ 3.086e+18                                # from Parsec to cm
            
            for l in range(len(radius_vector[:,0,0])):
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color='m',linewidth=0.3)
                
            ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
            ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
            
        zoom = np.max(radius_to_origin)
        
        ax.set_xlim(-zoom,zoom)
        ax.set_ylim(-zoom,zoom)
        ax.set_zlim(-zoom,zoom)
        
        ax.set_xlabel('x [Pc]')
        ax.set_ylabel('y [Pc]')
        ax.set_zlabel('z [Pc]')

        ax.set_title('Magnetic field morphology')
        
        plt.savefig(f"{output_path}/MagneticFieldTopology.png", bbox_inches='tight')
