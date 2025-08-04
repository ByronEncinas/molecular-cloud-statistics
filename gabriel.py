import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
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

# cache-ing spatial.cKDTree(Pos[:]).query(x, k=1)
_cached_tree = None
_cached_pos = None

def find_points_and_relative_positions(x, Pos, VoronoiPos):
    global _cached_tree, _cached_pos
    if _cached_tree is None or not np.array_equal(Pos, _cached_pos):
        _cached_tree = cKDTree(Pos)
        _cached_pos = Pos.copy()

    dist, cells = _cached_tree.query(x, k=1, workers=-1)
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

""" 
python3 los_stats.py 2000 ideal 430 50 N seed > NLOS430TST.txt 2> NLOS430TST_error.txt &

N : Column densities

"""
if len(sys.argv)>5:
    N                 = int(sys.argv[1])
    rloc              = float(sys.argv[2])
    case              = str(sys.argv[3]) #ideal/amb
    num_file          = str(sys.argv[4]) 
    max_cycles        = int(sys.argv[5]) 
else:
    N               = 3_000
    rloc            = 0.1
    case            = 'ideal'
    num_file        = '430'
    max_cycles      = 2000

ncrit = 10e2
if case == 'ideal':
    subdirectory = 'ideal_mhd'
elif case == 'amb':
    subdirectory = 'ambipolar_diffusion'
else:
    subdirectory= ''

file_list = glob.glob(f'arepo_data/{subdirectory}/*.hdf5')
filename = None

for f in file_list:
    if num_file in f:
        filename = f
if filename == None:
    raise FileNotFoundError

file_path = f'./{case}_cloud_trajectory.txt'

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

snap = filename.split(".")[0][-3:]

new_folder = os.path.join(f"los/{case}" , snap)
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

def uniform_in_3d(no, rloc=1.0, ncrit=100): # modify
    uniform_in_3d.c = 0
    def xyz_gen(size):
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U2 = np.random.uniform(low=0.0, high=1.0, size=size)
        U3 = np.random.uniform(low=0.0, high=1.0, size=size)
        r = rloc*np.cbrt(U1)
        theta = np.arccos(2*U2-1)
        phi = 2*np.pi*U3
        x,y,z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
        if uniform_in_3d.c == 0:                   
            bins = 100
            plt.hist(r, bins=bins, color = 'skyblue', density =True)
            plt.title('PDF $r = R\sqrt[3]{U(0,1)}$')
            plt.ylabel(r'PDF')
            plt.xlabel("$r$ (pc)")
            plt.grid()
            plt.tight_layout()
            plt.savefig('./images/pdf_r.png')
            plt.close()

            plt.hist(theta, bins=bins, color = 'skyblue', density =True)
            plt.title('PDF $\\theta = \\arccos(2U-1)$')
            plt.ylabel(r'PDF')
            plt.xlabel('$\\theta$ (rad)')
            plt.grid()
            plt.tight_layout()
            plt.savefig('./images/pdf_theta.png')
            plt.close()
        c=1
        return np.array([[a,b,c] for a,b,c in zip(x,y,z)])

    tree = cKDTree(Pos)
    valid_vectors = []
    rho_vector = np.zeros((no, 3))
    from copy import deepcopy
    while len(valid_vectors) < no:
        aux_vector = xyz_gen(no- len(valid_vectors)) # [[x,y,z], [x,y,z], ...] <= np array
        distances = np.linalg.norm(aux_vector, axis=1)
        inside_sphere = aux_vector[distances <= rloc]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > ncrit
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
        print(no, len(valid_vectors))
    rho_vector = np.array(deepcopy(valid_vectors))
    
    return rho_vector

if False: # Note to self, for small rloc < 1 this code doesnt take long
    rho = uniform_in_3d(100_000, rloc=0.25, ncrit=ncrit)
    print("rho_shape ", rho.shape)
    print(np.mean(rho[:,0]), np.mean(rho[:,1]), np.mean(rho[:,2]))
    print(np.std(rho[:,0]), np.std(rho[:,1]), np.std(rho[:,2]))
    print(np.median(rho[:,0]), np.median(rho[:,1]), np.median(rho[:,2]))
    
    if rho.shape[0]//100 > 1:
         bin = rho.shape[0]//100
    else:
        bin = 10
         
    plt.hist(rho[:,0], bins=bin, density=True)
    plt.title('PDF $x(n_g > n_crit)$')
    plt.ylabel(r'PDF')
    plt.xlabel('$x$ (pc)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./images/pdf_x.png')
    plt.close()

    plt.scatter(rho[:,0], rho[:,1], color = 'red', s=0.1)
    plt.grid()
    plt.tight_layout()
    plt.savefig('./images/xy_distro.png')
    plt.close()

    plt.scatter(rho[:,0], rho[:,2], color = 'red', s=0.1)
    plt.grid()
    plt.tight_layout()
    plt.savefig('./images/xz_distro.png')
    plt.close()

def line_of_sight(x_init=None, directions=fibonacci_sphere(), n_crit = 10e+2):
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    directions = directions/np.linalg.norm(directions, axis=1)[:, np.newaxis]
    dx = 0.5

    """
    Here you need to 
    directions = its repeated version 'm' times
    directions = np.tile(directions, m)
    x_init     = figure out how to repeat according to the example
    """
    m0 = x_init.shape[0]
    l0 = directions.shape[0]
    print(m0, l0)
    directions = np.tile(directions, (m0, 1))
    x_init = np.repeat(x_init, l0, axis=0)
    m = x_init.shape[0]
    l = directions.shape[0]
    print(m, l)
    """
    Now, a new feature that might speed the while loop, can be to double the size of all arrays
    and start calculating backwards and forwards simultaneously. This creates a more difficult condition
    for the 'mask', nevertheless, for a large array 'x_init' it may not be as different and it will definitely scale efficiently in parallel
    """

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    densities = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
    threshold = np.zeros((m,))
    threshold_rev = np.zeros((m,))
    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init 
    x = x_init.copy()
    dummy, _0, densities[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Density_grad, Pos, VoronoiPos)
    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3
    k=0
    x_rev = x_init.copy()
    dummy_rev, _00, densities_rev[0,:], cells_rev = dummy, _0, densities[0,:], cells
    vol_rev = Volume[cells_rev]
    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens_rev = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    k_rev=0
    mask  = dens > n_crit# 1 if not finished
    un_masked = np.logical_not(mask) # 1 if finished
    mask_rev = dens_rev > n_crit
    un_masked_rev = np.logical_not(mask_rev)
    
    while np.any(mask) or np.any(mask_rev): # 0 or 0 == 0 


        mask = dens > n_crit                # True if continue
        un_masked = np.logical_not(mask) # True if concluded
        mask_rev = dens_rev > n_crit               # True if continue
        un_masked_rev = np.logical_not(mask_rev) # True if concluded

        print(k, dens[:2], dens_rev[:2])

        _, bfield, dens, vol, ke, pressure = Heun_step(x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        
        pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens *= gr_cm3_to_nuclei_cm3
        
        vol[un_masked] = 0               # artifically make cell volume of finished lines equal to cero

        dx_vec = ((4 / 3) * vol / np.pi) ** (1 / 3)

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        x += dx_vec[:, np.newaxis] * directions

        line[k+1,:,:]    = x
        densities[k+1,:] = dens

        _, bfield_rev, dens_rev, vol_rev, ke_rev, pressure_rev = Heun_step(x_rev, -1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        
        pressure_rev *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens_rev *= gr_cm3_to_nuclei_cm3
        
        vol_rev[un_masked_rev] = 0 # unifinished lines will have cero volume for their corresponding cell

        dx_vec = ((4 / 3) * vol_rev / np.pi) ** (1 / 3)  # Increment step size

        threshold_rev += mask_rev.astype(int)  # Increment threshold count only for values still above 100

        x_rev -= dx_vec[:, np.newaxis] * directions

        line_rev[k+1,:,:]    = x_rev
        densities_rev[k+1,:] = dens_rev

        k_rev += 1
        k += 1
    
    threshold = threshold.astype(int)
    threshold_rev = threshold_rev.astype(int)

    max_th     = np.max(threshold) + 1
    max_th_rev = np.max(threshold_rev) + 1 

    radius_vector = np.append(line_rev[:max_th_rev,:,:][::-1, :, :], line[1:max_th,:,:], axis=0)*pc_to_cm
    numb_densities = np.append(densities_rev[:max_th_rev,:][::-1, :], densities[1:max_th,:], axis=0)
    column_densities = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vector, axis=0), axis=2), axis=0) # list of 'm' column densities
    
    print(column_densities.shape, l0, m0)

    average_columns = np.zeros(m0)
    i = 0
    for x in range(0, l0*m0, l0): # l0 corresponds with how many directions we have
        average_columns[i] = np.mean(column_densities[x:x+l0])
        print(i, x, x+l0, np.log10(average_columns[i]), column_densities[x:x+l0].shape)
        i += 1
    return radius_vector, numb_densities, average_columns

with open(os.path.join(new_folder, f'PARAMETERS'), 'w') as file:
    file.write(f"{sys.argv}\n")
    file.write(f"{filename}\n")
    file.write(f"{peak_den}\n")
    file.write(f"Cores Used: {os.cpu_count()}\n")
    file.write(f"Snap Time (Myr): {time_value}\n")
    file.write(f"rloc (Pc) : {rloc}\n")
    file.write(f"Density Threshold : {ncrit}\n")
    file.write(f"max_cycles         : {max_cycles}\n")
    file.write(f"Volume Range (Pc^3)   : ({Volume[np.argmin(Volume)]}, {Volume[np.argmax(Volume)]}) \n")
    file.write(f"Smallest Density (N/cm^3)  : ({Density[np.argmax(Volume)]*gr_cm3_to_nuclei_cm3},{Density[np.argmin(Volume)]*gr_cm3_to_nuclei_cm3}) \n")
    file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")

os.makedirs(new_folder, exist_ok=True)

# directions
directions = fibonacci_sphere(5)
print("No. of directions", directions.shape)

# starting positions 
print(max_cycles)
x_init = uniform_in_3d(max_cycles, rloc, ncrit)
#null_vector = np.array([[0.0, 0.0, 0.0]])  # shape (1, 3)
#x_init = np.vstack([x_init, null_vector]) 
print("No. of starting pos", x_init.shape)

radius_vector, numb_densities, average_column = line_of_sight(x_init, directions, ncrit)

os.makedirs(new_folder, exist_ok=True)
h5_path = os.path.join(new_folder, f"DataBundleG.h5")

# === Writing to HDF5 with metadata ===
with h5py.File(h5_path, "w") as f:
    # Datasets
    #f.create_dataset("positions", data=radius_vector)
    #f.create_dataset("densities", data=numb_densities)
    #f.create_dataset("magnetic_fields", data=magnetic_fields)
    f.create_dataset("starting_point", data=x_init)
    f.create_dataset("directions", data=directions)
    f.create_dataset("column_densities", data=average_column)

    # === Metadata attributes ===
    #f.attrs["seed"] = seed
    f.attrs["cores_used"] = os.cpu_count()
    f.attrs["pre_allocation_number"] = 2 * N
    f.attrs["rloc"] = rloc
    f.attrs["max_cycles"] = max_cycles
    f.attrs["center"] = Center
    f.attrs["volume_range"] = [Volume[np.argmin(Volume)],Volume[np.argmax(Volume)]]
    f.attrs["density_range"] = [float(gr_cm3_to_nuclei_cm3 * Density[np.argmin(Volume)]) ,float(gr_cm3_to_nuclei_cm3 * Density[np.argmax(Volume)])]
    
    try:
        pass
        #os.remove(tmp_file)
    except:
        print(f"Temp file not found tmp.csv")
        raise FileNotFoundError

print((time.time()-start_time)//60, " Minutes")

print("No. of averaged columns", average_column.shape)
print()

distance = np.linalg.norm(x_init, axis=1)*pc_to_cm

import matplotlib.pyplot as plt

if True:
    fig, ax = plt.subplots()
    
    # Axis labels
    ax.set_xlabel(r'''Distance $r$ (cm)
                
    Column density along line of sight as a function of distance 
    away from core
            $N_{LOS} = \int_0^s n_g(s')ds'$   
            ''')
    
    ax.set_ylabel(r'$N_{\rm LOS}$ (cm$^{-2}$)', fontsize=12)
    
    # Log scales
    ax.set_yscale('log')
    #ax.set_xscale('log')
    
    # Scatter plot with label for legend
    ax.scatter(distance, average_column, color='dodgerblue', s=2.0, label=r'$\langle N_{\rm LOS} \rangle$')
    
    # Ticks and grid
    ax.tick_params(axis='both')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Title and legend
    plt.title(r"$N_{\rm LOS} \propto r$", fontsize=13)
    ax.legend()
    
    # Layout and save
    fig.tight_layout()
    plt.savefig('./avg_column_los.png')
    plt.close()

# My goal here is to plot Column Densities along with corresponding Ionizations rate

if False:
    Neff = np.logspace(19, 27, 100)

    """  
    \mathcal{L} & \mathcal{H} Model: Protons

    """
    model_H = [1.001098610761e7, -4.231294690194e6,  7.921914432011e5,
            -8.623677095423e4,  6.015889127529e3, -2.789238383353e2,
            8.595814402406e0, -1.698029737474e-1, 1.951179287567e-3,
            -9.937499546711e-6
    ]


    model_L = [-3.331056497233e+6,  1.207744586503e+6,-1.913914106234e+5,
                1.731822350618e+4, -9.790557206178e+2, 3.543830893824e+1, 
            -8.034869454520e-1,  1.048808593086e-2,-6.188760100997e-5, 
                3.122820990797e-8]

    logzl = []
    for i,Ni in enumerate(Neff):
        lzl = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
        logzl.append(lzl)
    #print("Extremes of fitting zeta(N) ", logzl[0], logzl[-1])

    #logzetalfit = np.array(logzl)

    logzh = []

    for i,Ni in enumerate(Neff):
        lzh = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_H)] )
        logzh.append(lzh)

    from scipy import interpolate

    log_L = interpolate.interp1d(Neff, logzl)
    log_H = interpolate.interp1d(Neff, logzh)
    ionization_rates_l = log_L(column_densities)
    ionization_rates_h = log_H(column_densities) 
    plt.plot(Neff, log_H(Neff), label = 'Padovani H')
    plt.plot(Neff, log_L(Neff), label = 'Padovani L')
    plt.plot(column_densities, ionization_rates_l, label = '$N_{LOS}$ Model L')
    plt.plot(column_densities, ionization_rates_h, label = '$N_{LOS}$ Model H')
    plt.xscale('log')
    plt.legend()
    plt.savefig('./images/padovani2018')
    plt.close()


