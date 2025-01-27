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
    num_file         = '430'

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
directory_path = f"arepo_data/{typpe}/{subdirectory}"
file_list = glob.glob(f"{directory_path}/*.hdf5")

# Print the first 5 files for debugging/inspection
print(file_list[:5])

filename = None

for f in file_list:
    if num_file in f:
        filename = f

snap = filename.split(".")[0][-3:]

# Create the directory path
directory_path = os.path.join(f"density_profiles/{typpe}/", snap)

# Check if the directory exists, if not create it
if not os.path.exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    print(f"Directory {directory_path} created.")
else:
    print(f"Directory {directory_path} already exists.")

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
    temperature = (pressure * mu * m_H) / (mass_dens * boltzmann_constant_cgs)
    grav_potential = 0.0

    energy_magnetic[0,:] = bfields[0,:]*bfields[0,:]/(8*np.pi)*vol
    energy_thermal[0,:]  = (3 / 2) * pressure * (4*np.pi*rad**2*dx_vec)
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
        
        mass_dens *= code_units_to_gr_cm3
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
        M_r = np.cumsum(4 * np.pi * dens * rad ** 2 * dx_vec * parsec_to_cm3)

        # Gravitational potential: φ(r)
        phi = (grav_constant_cgs * M_r[-1] / rad[-1]) - np.cumsum(grav_constant_cgs * M_r / rad ** 2)

        # Gravitational energy
        grav_energy_density = 4 * np.pi * rad ** 2 * dens * phi

        energy_grav[k + 1, :]     = np.cumsum(grav_energy_density * dx_vec)
        energy_magnetic[k + 1, :] = energy_magnetic[k, :] +  bfield * bfield / (8 * np.pi) * (4*np.pi*rad**2*dx_vec)
        energy_thermal[k + 1, :]  = energy_thermal[k, :] + (3 / 2) * pressure * (4*np.pi*rad**2*dx_vec)

        # Effective column densities
        eff_column_densities[k + 1, :] = eff_column_densities[k, :] + dens * dx_vec

        if np.all(un_masked):
            print("All values are False: means all density < 10^2")
            break

        k += 1

    mean_column = np.mean(eff_column_densities[-1,:])
    ratio_thermal = energy_thermal / energy_grav
    ratio_magnetic = energy_magnetic / energy_grav

    threshold = threshold.astype(int)
    larger_cut = np.max(threshold)
    
    # cut all of them to standard
    radius_vector   = line[:larger_cut+1,:,:]
    magnetic_fields = bfields[:larger_cut+1,:]
    numb_densities   = densities[:larger_cut+1,:]
    volumes         = volumes[:larger_cut+1,:]

    # Initialize trajectory and radius_to_origin with the same shape
    trajectory      = np.zeros_like(magnetic_fields)
    radius_to_origin= np.zeros_like(magnetic_fields)

    print("Magnetic fields shape:", magnetic_fields.shape)
    print("Radius vector shape:", radius_vector.shape)
    print("Numb densities shape:", numb_densities.shape)

    for _n in range(m):  # Iterate over the first dimension

        prev = radius_vector[0, _n, :]
        
        for k in range(magnetic_fields.shape[0]):  # Iterate over the first dimension

            radius_to_origin[k, _n] = magnitude(radius_vector[k, _n, :])
            cur = radius_vector[k, _n, :]
            diff_rj_ri = magnitude(cur, prev)
            trajectory[k,_n] = trajectory[k-1,_n] + diff_rj_ri            
            prev = radius_vector[k, _n, :]

    radius_vector   *= 1.0* parsec_to_cm3
    trajectory      *= 1.0* parsec_to_cm3
    magnetic_fields *= 1.0* gauss_code_to_gauss_cgs
    trajectory[0,:]  = 0.0

    return radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, threshold, [mean_column, ratio_magnetic, ratio_thermal, energy_grav]

print("Steps in Simulation: ", N)
print("Boxsize            : ", Boxsize)
print("Smallest Volume    : ", Volume[np.argmin(Volume)])
print("Biggest  Volume    : ", Volume[np.argmax(Volume)])
print(f"Smallest Density  : {Density[np.argmin(Density)]}")
print(f"Biggest  Density  : {Density[np.argmax(Density)]}")

radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, threshold, col_energies = get_along_lines(x_init)

print("Elapsed Time: ", (time.time() - start_time)/60.)

import matplotlib.pyplot as plt

for i in range(m):

    mean_column, ratio_magnetic, ratio_thermal, energy_grav = col_energies
    
    cut = threshold[i]

    if True:
        # Define a mosaic layout for 5 plots (removed one for gravitational energy)
        mosaic = [
            ['A', 'B'],
            ['E', '.', '.'],  # This row now has 3 elements (same length as the first row)
            ['C', 'D']
        ]
        # Create a figure with the mosaic layout (3 rows, 3 columns)
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(10, 8), dpi=300)

        # Plot 1: Magnetic Energy
        axs['A'].plot(trajectory[:cut, i], magnetic_fields[:cut, i], linestyle="--", color="blue")
        axs['A'].scatter(trajectory[:cut, i], magnetic_fields[:cut, i], marker="o", color="blue", s=5)
        axs['A'].set_xlabel("s (cm) along LOS")
        axs['A'].set_ylabel("Energy (ergs)")
        axs['A'].set_title("Magnetic Energy (LOS)")
        axs['A'].grid(True)

        # Plot 2: Thermal Energy
        axs['B'].plot(trajectory[:cut, i], numb_densities[:cut, i], linestyle="--", color="red")
        axs['B'].scatter(trajectory[:cut, i], numb_densities[:cut, i], marker="o", color="red", s=5)
        axs['B'].set_xlabel("s (cm) along LOS")
        axs['B'].set_ylabel("Energy (ergs)")
        axs['B'].set_title("Thermal Energy (LOS)")
        axs['B'].grid(True)

        # Plot 3: Energies as a Fraction of Gravitational Energy
        axs['C'].plot(trajectory[:cut, i], ratio_magnetic[:cut, i], linestyle="--", color="blue", label="Magnetic/Gravitational (LOS)")
        axs['C'].plot(trajectory[:cut, i], ratio_thermal[:cut, i], linestyle="--", color="red", label="Thermal/Gravitational (LOS)")
        axs['C'].set_yscale('log')
        axs['C'].set_xlabel("s (cm) along LOS")
        axs['C'].set_ylabel("Energy Ratio")
        axs['C'].set_title("Energies Relative to Gravitational Energy (LOS)")
        axs['C'].legend()
        axs['C'].grid(True)

        # Plot 4: Energy Volume Ratios
        axs['D'].plot(trajectory[:cut, i], energy_grav[:cut,i], linestyle="--", color="orange", label="Grav Energy (LOS)")
        axs['D'].set_xlabel("s (cm) along LOS")
        axs['D'].set_ylabel("$E_{grav}$")
        axs['D'].set_title("Gravitational Energy (LOS)")
        axs['D'].legend()
        axs['D'].grid(True)

        # Add a table with summary data at the bottom of the figure
        table_data = [
            ['---', 'Value', 'Note'],
            ['Mean Column Density (LOS)', f'{mean_column:.3f}', '-'],
            ['Mean Magnetic/Thermal Ratio (LOS)', f'{ratio_magnetic.mean():.3f}', '-'],
            ['Steps in Simulation (LOS)', str(N), '-'],
            ['Smallest Volume (LOS)', f'{Volume[np.argmin(Volume)]:.3e}', '-'],
            ['Biggest Volume (LOS)', f'{Volume[np.argmax(Volume)]:.3e}', '-'],
            ['Smallest Density (LOS)', f'{Density[np.argmin(Density)]:.3e}', '-'],
            ['Biggest Density (LOS)', f'{Density[np.argmax(Density)]:.3e}', '-']
        ]
        table = axs['E'].table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])

        axs['E'].axis('off')  # Hide axis for the table plot

        # Adjust the layout to avoid overlap between plots
        plt.tight_layout()

        # Save the generated figure as a PNG file
        plt.savefig(f"{directory_path}/energies_mosaic_{i}.png", dpi=300)

        # Close the plot to release resources
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
        
        plt.savefig(f"{directory_path}/MagneticFieldTopology.png", bbox_inches='tight')
