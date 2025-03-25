from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from library import *
import csv, glob, os, sys, time
import h5py, json

start_time = time.time()

FloatType = np.float64
IntType = np.int32

if len(sys.argv)>4:
    #python3 arepo_reduction_factor_colors.py N rloc max_cycles num_file SLURM_JOB_ID
    #python3 arepo_reduction_factor_colors.py 500 0.5 100 430 
	N=int(sys.argv[1])
	rloc=float(sys.argv[2])
	max_cycles   =int(sys.argv[3])
	case = f'{sys.argv[4]}'
	num_file = f'{sys.argv[5]}'
	if len(sys.argv) < 6:
		sys.argv.append('NO_ID')
else:
    N            =5_000
    rloc=1   # rloc for boundary region of the cloud
    max_cycles   =500
    case = 'ideal'
    num_file = '430'

print(sys.argv)

cycle = 0 
reduction_factor_at_numb_density = defaultdict()
reduction_factor = []

if case == 'ideal':
    subdirectory = 'ideal_mhd'
elif case == 'amb':
    subdirectory = 'ambipolar_diffusion'
else:
    subdirectory= ''

file_path       = f'cloud_tracker_slices/{case}/{case}_cloud_trajectory.txt'

snap = []
time_value = []
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
    next(csv_reader)  # Skip the header row
    print('File opened successfully')

    for row in csv_reader:
        if num_file == str(row[0]):
            print("Match found!")
            print("Row:", row)  # Print the row for debugging
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])
            snap =str(row[0])
            time_value = float(row[1])
            peak_den =  float(row[5])
try:
    print(Center)
except:
    raise ValueError('Center is not defined')

snap_array = np.array(snap)
time_value_array = np.array(time_value)

import glob

file_list = glob.glob(f"arepo_data/{subdirectory}/*.hdf5")

filename = None
for f in file_list:
    if num_file in f:
        filename = f
if filename is None:
    raise FileNotFoundError

data = h5py.File(filename, 'r')
header_group = data['Header']
os.makedirs("threshold_reduction_test", exist_ok=True)
parent_folder = "threshold_reduction_test/"+ case 
children_folder = os.path.join(parent_folder, snap)
os.makedirs(children_folder, exist_ok=True)
Boxsize = data['Header'].attrs['BoxSize']
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))
Volume   = Mass/Density

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

CloudCord = Center.copy()

VoronoiPos-=Center
Pos-=Center

for dim in range(3):  # Loop over x, y, z
    pos_from_center = Pos[:, dim]
    boundary_mask = pos_from_center > Boxsize / 2
    Pos[boundary_mask, dim] -= Boxsize
    VoronoiPos[boundary_mask, dim] -= Boxsize

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

    # Initialize trajectory and radius_to_origin with the same shape
    trajectory = np.zeros_like(magnetic_fields)
    radius_to_origin = np.zeros_like(magnetic_fields)
    column = np.zeros_like(magnetic_fields)

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
            column[k, _n] = column[k-1, _n] + numb_densities[k, _n] * diff_rj_ri            
            
            prev = cur  # Store current point as previous point

    radius_vector   *= 1.0#* 3.086e+18                                # from Parsec to cm
    trajectory      *= 1.0#* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0#* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)
    volumes_all     *= 1.0#/(3.086e+18**3) 

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

x_init = generate_vectors_in_core(max_cycles, max_cycles, rloc)

print("Cores Used          : ", os.cpu_count())
print("Steps in Simulation : ", 2*N)
print("rloc                : ", rloc)
print("max_cycles          : ", max_cycles)
print("Boxsize             : ", Boxsize) # 256
print("Center              : ", Center) # 256
print("Posit Max Density   : ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume     : ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume     : ", Volume[np.argmax(Volume)]) # 256
print(f"Smallest Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmax(Volume)]}")
print(f"Biggest  Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmin(Volume)]}")

import concurrent.futures
from scipy.ndimage import gaussian_filter1d

test_thresh = [100, 10]

def process_case(case):
    radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th, cd = get_along_lines(x_init, case)
    return radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th

# Use ProcessPoolExecutor to parallelize the loop
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(process_case, test_thresh))

# The 'results' will contain the return values from the 'process_case' function for each case
for i, result in enumerate(results):
    radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th, cd = result

    case = test_thresh[i]

    #radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th = get_along_lines(x_init, case)

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

        ds = np.diff(distance) 
        adaptive_sigma = 3*ds/np.mean(ds) #(ds > np.mean(ds))
        bfield = np.array([gaussian_filter1d(bfield, sigma=s)[i] for i, s in enumerate(adaptive_sigma)])
        
        bfield = bfield[1:]
        distance = distance[1:]
        numb_density = numb_density[1:]

        pocket, global_info = smooth_pocket_finder(bfield, cycle, plot=False) # this plots
        index_pocket, field_pocket = pocket[0], pocket[1]

        min_den_cycle.append(min(numb_density))
        
        globalmax_index = global_info[0]
        globalmax_field = global_info[1]

        x_r = distance[p_r]
        B_r = bfield[p_r]
        n_r = numb_density[p_r]

        p_i = find_insertion_point(index_pocket, p_r)
        
        print("Maxima Values related to pockets: ", len(index_pocket), p_i)

        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
            B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
            success = True  
        except:
            R = 1
            reduction_factor.append(R)
            numb_density_at.append(n_r)
            pos_red[tupi] = R
            success = False 
            continue
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
        file.write(f"rloc (Pc) : {rloc}\n")
        file.write(f"x_init (Pc)        :\n {x_init}\n")
        file.write(f"max_cycles         : {max_cycles}\n")
        file.write(f"Boxsize (Pc)       : {Boxsize} Pc\n")
        file.write(f"Center (Pc, Pc, Pc): {CloudCord[0]}, {CloudCord[1]}, {CloudCord[2]} \n")
        file.write(f"Posit Max Density (Pc, Pc, Pc): {Pos[np.argmax(Density), :]}\n")
        file.write(f"Smallest Volume (Pc^3)   : {Volume[np.argmin(Volume)]} \n")
        file.write(f"Biggest  Volume (Pc^3)   : {Volume[np.argmax(Volume)]}\n")
        file.write(f"Smallest Density (M☉/Pc^3)  : {Density[np.argmax(Volume)]} \n")
        file.write(f"Biggest  Density (M☉/Pc^3) : {Density[np.argmin(Volume)]}\n")
        file.write(f"Smallest Density (N/cm^3)  : {np.log10(Density[np.argmax(Volume)]*gr_cm3_to_nuclei_cm3)} \n")
        file.write(f"Biggest  Density (N/cm^3) : {np.log10(Density[np.argmin(Volume)]*gr_cm3_to_nuclei_cm3)}\n")
        file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")

    print(f"Elapsed time: {(time.time() - start_time)/60.} Minutes")

    counter = Counter(reduction_factor)
    pos_red = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pos_red.items()}

    file_path = os.path.join(children_folder, f'reduction_factor_{case}_{sys.argv[-1]}.json')
    with open(file_path, 'w') as json_file: json.dump(reduction_factor, json_file)

    file_path = os.path.join(children_folder, f'numb_density_{case}_{sys.argv[-1]}.json')
    with open(file_path, 'w') as json_file: json.dump(numb_density_at, json_file)

    file_path = os.path.join(children_folder, f'position_vector_{case}_{sys.argv[-1]}')
    with open(file_path, 'w') as json_file: json.dump(pos_red, json_file)

    """# Graphs"""

    reduction_factor = reduction_factor[reduction_factor != 1.0]
    numb_density_at = numb_density_at[reduction_factor != 1.0]

    bins=len(reduction_factor)//10 
    if bins == 0:
        raise ValueError("No valid data found: bins cannot be zero.")

    total = len(reduction_factor)
    ones = counter['1']

    reduction_factor = np.array(reduction_factor)
    reduction_factor = reduction_factor[reduction_factor != 1.0]
    fraction = ones / total
    inverse_reduction_factor = np.array([1/reduction_factor[i] for i in range(len(reduction_factor))])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].hist(reduction_factor, bins=bins, color='skyblue', edgecolor='black', density=True)
    axs[0].set_yscale('log')
    axs[0].set_title(f'Distribution of R for $R \\neq 1$ (fraction = {fraction})')
    axs[0].set_xlabel(f'Reduction factor ({fraction})')
    axs[0].set_ylabel('PDF')

    axs[1].hist(inverse_reduction_factor, bins=bins, color='skyblue', edgecolor='black', density=True)
    axs[1].set_yscale('log')
    axs[1].set_title(f'Distribution of 1/R for $R \\neq 1$ (fraction = {fraction})')
    axs[1].set_xlabel(f'Reduction factor ')
    axs[1].set_ylabel('PDF')

    plt.tight_layout()


    plt.savefig(os.path.join(children_folder,f"hist_nth={case}.png"))