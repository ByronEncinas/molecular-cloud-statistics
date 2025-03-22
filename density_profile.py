
import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt, healpy as hp
from library import *

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
    pressure = Pressure[cells]
    
    return x_final, abs_local_fields_1, local_densities, CellVol, kinetic_energy, pressure

FloatType = np.float64
IntType = np.int32

if len(sys.argv)>2:
    N             = int(sys.argv[1])
    case         = str(sys.argv[2]) #ideal/amb
    num_file          = str(sys.argv[3]) 
else:
    N            = 4_000
    case        = 'ideal'
    num_file     = '430'

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

file_path = f'cloud_tracker_slices/{case}/{case}_cloud_trajectory.txt'

snap = []
time_value = []

with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        snap.append(int(row[0]))  # First column is snap
        time_value.append(float(row[1]))  # Second column is time_value
        if num_file == str(row[0]):
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])

snap_array = np.array(snap)
time_value_array = np.array(time_value)

file_list = glob.glob(f'arepo_data/{subdirectory}/*.hdf5')
filename = None

for f in file_list:
    if num_file in f:
        filename = f
if filename == None:
    raise FileNotFoundError

snap = filename.split(".")[0][-3:]

new_folder = os.path.join(f"./density_profiles/{case}/", num_file)
os.makedirs(new_folder, exist_ok=True)

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Pressure = np.asarray(data['PartType0']['Pressure'], dtype=FloatType)
Velocities = np.asarray(data['PartType0']['Velocities'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))
Volume   = Mass/Density

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

file_path       = f'cloud_tracker_slices/{case}/{case}_cloud_trajectory.txt'

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

init_files = glob.glob(f'./x_init_100.npy')

if init_files:
    x_init = np.load(init_files[0], mmap_mode='r')
else:
    pass

try:
    # for column densities propagated in field direction from several starting points
    directions, abs_local_fields, local_densities, _ = find_points_and_get_fields(x_init, Bfield, Density, Density_grad, Pos, VoronoiPos)
    print('Directions provided by B field at point')
except:
    # for line os sight with start in center
    directions = fibonacci_sphere(100)
    m = directions.shape[0]
    x_init = np.zeros((m,3))
     
new_folder = os.path.join("density_profiles/" , snap)
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
    grav_potential = 0.0

    energy_magnetic[0,:] = bfields[0,:]*bfields[0,:]*(gauss_code_to_gauss_cgs)**2/(8*np.pi)*(vol*parsec_to_cm3**3)
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
        _, bfield, dens, vol, ke, pressure = Heun_step(
            x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
        )
        
        mass_dens = dens * code_units_to_gr_cm3
        pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens *= gr_cm3_to_nuclei_cm3
        
        vol[un_masked] = 0

        non_zero = vol > 0
        if len(vol[non_zero]) == 0:
            break

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

        binding_energy = -np.sum((grav_constant_cgs * M_r / (rad*parsec_to_cm3)) * 4 * np.pi * (rad*parsec_to_cm3)**2 * mass_dens * dx_vec*parsec_to_cm3)

        energy_grav[k + 1, :]     = binding_energy
        energy_magnetic[k + 1, :] = energy_magnetic[k, :] +  bfield * bfield*(gauss_code_to_gauss_cgs)**2 / (8 * np.pi) * (4*np.pi*(rad*parsec_to_cm3)**2*(dx_vec*parsec_to_cm3))
        energy_thermal[k + 1, :]  = energy_thermal[k, :] + (3 / 2) * pressure * (4*np.pi*(rad*parsec_to_cm3))**2*(dx_vec*parsec_to_cm3)

        eff_column_densities[k + 1, :] = eff_column_densities[k, :] + dens * (dx_vec*parsec_to_cm3)

        print(f"Eff. Column Densities: {eff_column_densities[k + 1, 0]:5e}")

        if np.all(un_masked):
            print("All values are False: means all density < 10^2")
            break

        k += 1

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

os.makedirs(new_folder, exist_ok=True)

if True:
    for i in range(m):

        eff_column_densities, energy_magnetic, energy_thermal, energy_grav = col_energies
        cut = threshold[i]
        eff_column = np.max(eff_column_densities[:, i])

        order_total_energy = np.log10(energy_magnetic[:cut, i] + energy_thermal[:cut, i] + energy_grav[:cut, i])
        print(order_total_energy)

        np.save(f"{new_folder}/eff_column_densities_{i}.npy", eff_column_densities[:cut, i])
        np.save(f"{new_folder}/numb_densities_{i}.npy", numb_densities[:cut, i])
        np.save(f"{new_folder}/energy_magnetic_{i}.npy", energy_magnetic[:cut, i])
        np.save(f"{new_folder}/energy_thermal_{i}.npy", energy_thermal[:cut, i])
        np.save(f"{new_folder}/energy_grav_{i}.npy", energy_grav[:cut, i])

        # Define new mosaic layout
        mosaic = [
            ['A', 'B'],
            ['C', 'C']
        ]
        fig, axs = plt.subplot_mosaic(mosaic, figsize=(12, 10), dpi=300)

        # Plot Number Density
        axs['A'].plot(trajectory[1:cut, i], numb_densities[1:cut, i], linestyle="--", color="blue")
        axs['A'].set_yscale('log')
        axs['A'].set_xscale('log')
        axs['A'].set_xlabel("s (cm) along LOS")
        axs['A'].set_ylabel("Number Density $n_g(s)$")
        axs['A'].set_title("Number Density Along LOS")
        axs['A'].grid(True)

        # Plot Energy Ratios
        axs['B'].plot(trajectory[1:cut, i], energy_magnetic[1:cut, i] / abs(energy_grav[1:cut, i]), linestyle="--", color="red", label="Magnetic / Gravity")
        axs['B'].plot(trajectory[1:cut, i], energy_thermal[1:cut, i] / abs(energy_grav[1:cut, i]), linestyle="--", color="green", label="Thermal / Gravity")
        axs['B'].set_xscale('log')
        axs['B'].set_yscale('log')
        axs['B'].set_xlabel("s (cm) along LOS")
        axs['B'].set_ylabel("Energy Ratios")
        axs['B'].set_title("Energy Ratios Along Line of Sight")
        axs['B'].legend()
        axs['B'].grid(True)

        # Table Data
        table_data = [
            ['---', 'Value', 'Note'],
            ['Column Density (LOS)', f'{eff_column:.5e}', '-'],
            ['Steps in Simulation (LOS)', str(len(trajectory)), '-'],
            ['Smallest Volume (LOS)', f'{np.max(volumes[:cut, i]):.3e}', '-'],
            ['Biggest Volume (LOS)', f'{np.max(volumes[:cut, i]):.3e}', '-'],
            ['Smallest Density (LOS)', f'{np.min(numb_densities[:cut, i]):.3e}', '-'],
            ['Biggest Density (LOS)', f'{np.max(numb_densities[:cut, i]):.3e}', '-']
        ]
        table = axs['C'].table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
        axs['C'].axis('off')

        np.savetxt(f"{new_folder}/table_{i}.txt", table_data, fmt="%s", delimiter="   ")
        plt.tight_layout()
        plt.savefig(f"{new_folder}/energy_ratios{i}.png", dpi=300)
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

if False:
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
        axs['A'].set_xscale('log')
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
