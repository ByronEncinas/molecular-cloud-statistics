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
else:
    N            = 4_000

"""  B. Jesus Velazquez    """

directory_path = "arepo_data/ideal_mhd/"
files = list_files(directory_path, '.hdf5')

total_files = len(files)
num_to_pick = 10

# Generate 20 evenly spaced indices
indices = np.linspace(0, total_files - 1, num_to_pick, dtype=int)

# Select the files using the generated indices
selected_files = [files[i] for i in indices]

# Print the selected files
print(selected_files)

for filename in files:
    snap = filename.split(".")[0][-3:]

    # Create the directory path
    directory_path = os.path.join("density_profiles", snap)

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

    Center = Pos[np.argmax(Density),:] #430
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
        ie = InternalEnergy[cells]
        ke = 0.5*Mass[cells]*np.linalg.norm(Velocities[cells], axis=1)**2
        pressure = Pressure[cells]* mass_unit / (length_unit * (time_unit ** 2))  #cgs
        molecular_weight = 1 # for atomic hydrogen
        temp = (pressure/mass_dens)*molecular_weight*boltzmann_constant_cgs
        mass = 0.0

        mass_shell_cumulative = 4 * np.pi * dens * (rad**2) * (dx_vec * parsec_to_cm3) * parsec_to_cm3
        phi_grav = grav_constant_cgs*Mass
        grav_potential = -(4*np.pi)**2 * (dens**2) * (rad*parsec_to_cm3)**4*(dx_vec*parsec_to_cm3)
        
        energy_magnetic[0,:] = bfields[0,:]*bfields[0,:]/(8*np.pi)
        energy_thermal[0,:]  = (3/2)*boltzmann_constant_cgs*temp* (4*np.pi*dens*vol)
        energy_grav[0,:]     = grav_potential
        
        k=0

        mask = dens > 100 # True if continue
        un_masked = np.logical_not(mask)

        while np.any((mask)):

            # Create a mask for values that are 10^2 N/cm^3 above the threshold
            mask = dens > 100 # True if continue
            un_masked = np.logical_not(mask)
            
            # Only update values where mask is True
            _, bfield, dens, vol, ke, ie, pressure = Heun_step(x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)

            mass_dens *=code_units_to_gr_cm3
            pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
            dens *= gr_cm3_to_nuclei_cm3 
            
            vol[un_masked] = 0

            non_zero = vol > 0
            if len(vol[non_zero]) == 0:
                break

            dx_vec = np.min(((4 / 3) * vol[non_zero] / np.pi) ** (1 / 3)) # make sure to cover the shell in all directions at the same pace

            threshold += mask.astype(int)  # Increment threshold count only for values still above 100

            x += dx_vec * directions

            line[k+1,:,:]    = x
            volumes[k+1,:]   = vol
            bfields[k+1,:]   = bfield
            densities[k+1,:] = dens

            # Line of Sight 
            rad      = np.linalg.norm(x[:,:], axis=1)
            surface_area = 4*np.pi*rad**2
            avg_den  = np.mean(mass_dens)
            
            mass_shell_cumulative += 4*np.pi*dens*(rad*rad*parsec_to_cm3*parsec_to_cm3)*(dx_vec*parsec_to_cm3) # adding concentric shells with density 'dens'
            mass += 4*np.pi*np.average(dens)*(rad*rad*parsec_to_cm3*parsec_to_cm3)*(dx_vec*parsec_to_cm3)
            grav_potential += -(4*np.pi)**2 * (dens**2) * (rad*parsec_to_cm3)**4*(dx_vec*parsec_to_cm3)

            energy_grav[k+1,:] = grav_potential
            energy_magnetic[k+1,:] = bfield*bfield/(8*np.pi)*vol 
            energy_thermal[k+1,:] = (3/2)*boltzmann_constant_cgs*temp
            eff_column_densities[k+1,:] = dens*dx_vec

            if np.all(un_masked):
                print("All values are False: means all density < 10^2")
                break

            k += 1

        mean_column = np.mean(eff_column_densities[-1,:])

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

            cut = threshold[_n] - 1

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

        return radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, threshold, [energy_grav, energy_magnetic, energy_thermal]

    print("Steps in Simulation: ", N)
    print("Boxsize            : ", Boxsize)
    print("Smallest Volume    : ", Volume[np.argmin(Volume)])
    print("Biggest  Volume    : ", Volume[np.argmax(Volume)])
    print(f"Smallest Density  : {Density[np.argmin(Density)]}")
    print(f"Biggest  Density  : {Density[np.argmax(Density)]}")

    radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, threshold, energies = get_along_lines(x_init)

    print("Elapsed Time: ", (time.time() - start_time)/60.)

    for i in range(m):

        energy_grav, energy_magnetic, energy_thermal = energies
        
        cut = threshold[i]

    if True:
        # Define a mosaic layout for 6 plots
        mosaic = """
        ABC
        DEF
        """
        
        # Create a figure with the mosaic layout
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(10, 8))
        
        # Energy plot 1: Gravitational Energy
        axs['A'].plot(trajectory[:cut, i], energy_grav[:cut, i], linestyle="--", color="green")
        axs['A'].scatter(trajectory[:cut, i], energy_grav[:cut, i], marker="o", color="green", s=10)
        axs['A'].set_xlabel("s (cm)")
        axs['A'].set_ylabel("Energy (ergs)")
        axs['A'].set_title("Gravitational Energy")
        axs['A'].grid(True)
        
        # Energy plot 2: Magnetic Energy
        axs['B'].plot(trajectory[:cut, i], energy_magnetic[:cut, i], linestyle="--", color="blue")
        axs['B'].scatter(trajectory[:cut, i], energy_magnetic[:cut, i], marker="o", color="blue", s=10)
        axs['B'].set_xlabel("s (cm)")
        axs['B'].set_ylabel("Energy (ergs)")
        axs['B'].set_title("Magnetic Energy")
        axs['B'].grid(True)
        
        # Energy plot 3: Thermal Energy
        axs['C'].plot(trajectory[:cut, i], energy_thermal[:cut, i], linestyle="--", color="red")
        axs['C'].scatter(trajectory[:cut, i], energy_thermal[:cut, i], marker="o", color="red", s=10)
        axs['C'].set_xlabel("s (cm)")
        axs['C'].set_ylabel("Energy (ergs)")
        axs['C'].set_title("Thermal Energy")
        axs['C'].grid(True)
        
        # Energy plot 4: Combined Energies
        axs['D'].plot(trajectory[:cut, i], energy_grav[:cut, i], linestyle="--", color="green", label="Gravitational")
        axs['D'].plot(trajectory[:cut, i], energy_magnetic[:cut, i], linestyle="--", color="blue", label="Magnetic")
        axs['D'].plot(trajectory[:cut, i], energy_thermal[:cut, i], linestyle="--", color="red", label="Thermal")
        axs['D'].set_yscale('log')
        axs['D'].set_xlabel("s (cm)")
        axs['D'].set_ylabel("Energy (ergs)")
        axs['D'].set_title("Log Scale Energies")
        axs['D'].legend()
        axs['D'].grid(True)
        
        # Energy plot 5: Energy Ratio 1
        axs['E'].plot(trajectory[:cut, i], energy_grav[:cut, i] / energy_thermal[:cut, i], linestyle="--", color="purple")
        axs['E'].set_xlabel("s (cm)")
        axs['E'].set_ylabel("Grav/Thermal Ratio")
        axs['E'].set_title("Gravitational vs Thermal")
        axs['E'].grid(True)
        
        # Energy plot 6: Energy Ratio 2
        axs['F'].plot(trajectory[:cut, i], energy_magnetic[:cut, i] / energy_thermal[:cut, i], linestyle="--", color="orange")
        axs['F'].set_xlabel("s (cm)")
        axs['F'].set_ylabel("Magnetic/Thermal Ratio")
        axs['F'].set_title("Magnetic vs Thermal")
        axs['F'].grid(True)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{directory_path}/energies_mosaic_{i}.png")

        # Close the plot
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
