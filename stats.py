from scipy.ndimage import gaussian_filter1d
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from library import *
import csv, glob, os, sys, time
import h5py, json

# python3 stats.py 1000 10 10 ideal 430 TEST seed > R430TST.txt 2> R430TST_error.txt &

start_time = time.time()

FloatType = np.float64
IntType = np.int32
if len(sys.argv)>5:
    N             = int(sys.argv[1])
    rloc          = float(sys.argv[2])
    max_cycles    = int(sys.argv[3]) 
    case          = str(sys.argv[4]) 
    num_file      = str(sys.argv[5]) 
    seed          = int(sys.argv[6])
else:
    N               = 2_000
    rloc            = 1.0
    max_cycles      = 50
    case            = 'ideal'
    num_file        = '430'
    seed            = 12345 
    sys.argv.append(seed)

print(sys.argv, N)

reduction_factor_at_numb_density = defaultdict()
reduction_factor = []

if case == 'ideal':
    subdirectory = 'ideal_mhd'
elif case == 'amb':
    subdirectory = 'ambipolar_diffusion'
else:
    subdirectory= ''

file_path       = f'./{case}_cloud_trajectory.txt'

snap = []
time_value = []
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
    next(csv_reader)  # Skip the header row
    print('File opened successfully')

    for row in csv_reader:
        if num_file == str(row[0]):
            print("Match found!")
            Center = np.array([float(row[2]),float(row[3]),float(row[4])])
            snap =str(row[0])
            time_value = float(row[1])
            peak_den =  float(row[5])
try:
    Center
except:
    raise ValueError('Center is not defined')

snap_array = np.array(snap)
time_value_array = np.array(time_value)

import glob

file_list = glob.glob(f"arepo_data/{subdirectory}/*.hdf5")

filename = None
for f in file_list:
    if num_file in f:
        print()
        filename = f
if filename is None:
    raise FileNotFoundError

data = h5py.File(filename, 'r')
header_group = data['Header']
os.makedirs("stats", exist_ok=True)
parent_folder = "thesis_stats/"+ case 
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
CloudCord = Center.copy()
VoronoiPos-=Center
Pos-=Center

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


for dim in range(3):  # Loop over x, y, z
    pos_from_center = Pos[:, dim]
    boundary_mask = pos_from_center > Boxsize / 2
    Pos[boundary_mask, dim] -= Boxsize
    VoronoiPos[boundary_mask, dim] -= Boxsize

print("Allocation Number: ", N)

def get_along_lines(x_init=None, N=N):
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    dx = 0.5

    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))
    volumes   = np.zeros((N+1,m))
    threshold = np.zeros((m,)).astype(int) # one value for each
    threshold2 = np.zeros((m,)).astype(int) # one value for each

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
    volumes_rev   = np.zeros((N+1,m))
    threshold_rev = np.zeros((m,)).astype(int) # one value for each
    threshold2_rev = np.zeros((m,)).astype(int) # one value for each

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init 
    
    x = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]

    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    k=0

    mask_rev = dens > 10
    un_masked_rev = np.logical_not(mask_rev)
    mask2_rev = dens > 100
    un_masked2_rev = np.logical_not(mask2_rev)

    repeat_rev = False
    while np.any((mask_rev)):

        mask_rev = dens > 10
        un_masked_rev = np.logical_not(mask_rev)
        mask2_rev = dens > 100
        un_masked2_rev = np.logical_not(mask2_rev)

        aux =  x[un_masked_rev]
        aux2 = x[un_masked2_rev]

        x, bfield, dens, vol = Heun_step(x, -dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold_rev += mask_rev.astype(int)
        threshold2_rev += mask2_rev.astype(int)

        x[un_masked_rev] = aux
        print(np.max(np.log10(dens)))

        if k + 1 >= N:
            if np.all(un_masked2_rev):
                print("All values are False: means all density < 10^2")
                break
            if repeat_rev == True:
                x_init_not_finished = x_init[un_masked_rev] # keep un-finished lines
                x_init              = x_init[mask_rev] # keep finished lines
                line_rev_not_finished = line[:,un_masked_rev,:]
                volumes_rev_not_finished = volumes[:,un_masked_rev]
                bfields_rev_not_finished = bfields[:,un_masked_rev]
                densities_rev_not_finished = densities[:,un_masked_rev]

                line_rev = line_rev[:,mask_rev,:]
                volumes_rev = volumes_rev[:,mask_rev]
                bfields_rev = bfields_rev[:,mask_rev]
                densities_rev = densities_rev[:,mask_rev]
                break
            auxlines = line_rev
            auxvolumes = volumes_rev
            auxbfields = bfields_rev
            auxdensities = densities_rev

            N_old = N+1
            N *= 2

            line_rev      = np.zeros((N, m, 3))
            bfields_rev   = np.zeros((N, m))
            densities_rev = np.zeros((N, m))
            volumes_rev   = np.zeros((N, m))

            line_rev[:N_old,:,:]    = auxlines
            volumes_rev[:N_old,:]   = auxvolumes
            bfields_rev[:N_old,:]   = auxbfields
            densities_rev[:N_old,:] = auxdensities

            repeat_rev = True

        line_rev[k+1,:,:] = x
        volumes_rev[k+1,:] = vol
        bfields_rev[k+1,:] = bfield
        densities_rev[k+1,:] = dens                  

        if np.all(un_masked_rev):
            print("All values are False: means all density < 10")
            break

        k += 1

    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3
    k=0

    mask  = dens > 10# 1 if not finished
    un_masked = np.logical_not(mask) # 1 if finished

    mask2 = dens > 100
    un_masked2 = np.logical_not(mask2) # 1 if finished

    while np.any(mask):

        # Create a mask for values that are 10^2 N/cm^3 above the threshold
        mask  = dens > 10 # 1 if not finished
        un_masked = np.logical_not(mask) # 1 if finished

        mask2 = dens > 100
        un_masked2 = np.logical_not(mask2) # 1 if finished

        aux = x[un_masked]
        aux2 =x[un_masked2]

        x, bfield, dens, vol = Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold  += mask.astype(int)  # Increment threshold count only for values still above 100
        threshold2 += mask2.astype(int)  # Increment threshold count only for values still above 100
      
        x[un_masked] = aux

        if k + 1 >= N:
            # if we go over the size of the array, we check if the density threshold 100cm-3 is reached
            # if not, then resize
            if np.all(un_masked2):
                print("All values are False: means all density < 10^2")
                break
            if repeat == True:
                x_init_not_finished = x_init[un_masked] # keep un-finished lines
                x_init              = x_init[mask]      # keep finished lines
                line_not_finished = line[:,un_masked,:]
                volumes_not_finished = volumes[:,un_masked]
                bfields_not_finished = bfields[:,un_masked]
                densities_not_finished = densities[:,un_masked]

                line = line[:,mask,:]
                volumes = volumes[:,mask]
                bfields = bfields[:,mask]
                densities = densities[:,mask]
                break
            auxlines = line
            auxvolumes = volumes
            auxbfields = bfields
            auxdensities = densities

            N_old = N+1
            N *= 2

            line      = np.zeros((N, m, 3))
            bfields   = np.zeros((N, m))
            densities = np.zeros((N, m))
            volumes   = np.zeros((N, m))

            line[:N_old,:,:]    = auxlines
            volumes[:N_old,:]   = auxvolumes
            bfields[:N_old,:]   = auxbfields
            densities[:N_old,:] = auxdensities

            repeat = True


        line[k + 1, :, :]      = x
        volumes[k + 1, :]      = vol
        bfields[k + 1, :]      = bfield
        densities[k + 1, :]    = dens

        if np.all(un_masked):
            print("All values are False: means all density < 10")
            break

        k += 1
    

    if repeat_rev and repeat:
        unfinished_forward = np.logical_not(mask)
        unfinished_reverse = np.logical_not(mask_rev)
        unfinished_total = np.logical_or(unfinished_forward, unfinished_reverse)
        x_init_unfinished = x_init[unfinished_total]


        np.savez(os.path.join(children_folder, f"uDataBundle{seed}.npz"),
        u_seed=seed,
        u_x_init=x_init_unfinished,
        u_line_rev=line_rev_not_finished,
        u_line=line_not_finished,
        u_volumes_rev=volumes_rev_not_finished,
        u_volumes=volumes_not_finished,
        u_bfields_rev=bfields_rev_not_finished,
        u_bfields=bfields_not_finished,
        u_densities_rev=densities_rev_not_finished,
        u_densities=densities_not_finished,
        )


    threshold = threshold.astype(int)

    updated_mask = np.logical_not(np.logical_and(mask, mask_rev))

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

    return radius_vector, trajectory, magnetic_fields, numb_densities, volumes_all, radius_to_origin, [threshold, threshold2, threshold_rev, threshold2_rev], column

def generate_vectors_in_core(max_cycles, densthresh, rloc=0.1, seed=12345):
    import numpy as np
    from scipy.spatial import cKDTree
    np.random.seed(seed)
    valid_vectors = []
    tree = cKDTree(Pos)
    while len(valid_vectors) < max_cycles:
        points = np.random.uniform(low=-rloc, high=rloc, size=(max_cycles, 3))
        distances = np.linalg.norm(points, axis=1)
        inside_sphere = points[distances <= rloc]
        _, nearest_indices = tree.query(inside_sphere, workers=-1)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > densthresh
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
    valid_vectors = np.array(valid_vectors)
    random_indices = np.random.choice(len(valid_vectors), max_cycles, replace=False)
    return valid_vectors[random_indices]

x_init = generate_vectors_in_core(max_cycles, 100, rloc, seed)

radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin, th, cd = get_along_lines(x_init)

m = magnetic_fields.shape[1]

# th 10  , th2 100   , th_rev 10    , th2_rev 100
threshold, threshold2, threshold_rev, threshold2_rev = th 

reduction_factor = list()
numb_density_at  = list()

min_den_cycle = list()
pos_red = dict()


np.savez(os.path.join(children_folder, f"DataBundle{seed}.npz"),
         column_density=cd,
         positions=radius_vector,
         trajectory=trajectory,
         number_densities=numb_densities,
         magnetic_fields=magnetic_fields,
         thresholds=np.array(th),
         starting_point=x_init
         )


N = threshold.shape[0] # number of lines
for cycle in range(max_cycles): # 10

    _from = N +1 - threshold_rev[cycle]
    _to   = N+ 1 + threshold[cycle] -1 # we sliced line[1:,:,:] so -1
    p_r = N + 1 - _from

    vector = radius_vector[_from:_to,cycle, :].copy()
    column = cd[_from:_to,cycle].copy()
    bfield    = magnetic_fields[_from:_to,cycle].copy()
    distance = trajectory[_from:_to,cycle].copy()
    numb_density = numb_densities[_from:_to,cycle].copy()
    tupi = f"{x_init[cycle,0]},{x_init[cycle,1]},{x_init[cycle,2]}"

    print(len(bfield), p_r, threshold[cycle], threshold_rev[cycle])

    if bfield.shape[0] == 0:
        R = 1
        reduction_factor.append(R)
        numb_density_at.append(n_r)
        pos_red[tupi] = R
        continue
        
    if False:
        inter = (abs(np.roll(distance, 1) - distance) != 0) # removing pivot point
        distance = distance[inter]
        ds = np.abs(np.diff(distance, n=1))
        distance = distance[:-1]  
        vector = vector[inter][:-1]
        numb_density  = numb_density[inter][:-1]
        column        = column[inter][:-1]
        bfield        = bfield[inter]

        adaptive_sigma = 3*ds/np.mean(ds) #(ds > np.mean(ds))
        adaptive_sigma[adaptive_sigma==0] = 1.0e-1
        bfield = np.array([gaussian_filter1d(bfield, sigma=s)[i] for i, s in enumerate(adaptive_sigma)]) # adapatative stepsize impact extremes (cell size dependent)

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


counter = Counter(reduction_factor)

pos_red = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pos_red.items()}

with open(os.path.join(children_folder, f'PARAMETER_reduction10_{seed}'), 'w') as file:
    file.write(f"{filename}\n")
    file.write(f"{peak_den}\n")
    file.write(f"Run ID/seed: {seed}\n")
    file.write(f"Cores Used: {os.cpu_count()}\n")
    file.write(f"Snap Time (Myr): {time_value}\n")
    file.write(f"rloc (Pc) : {rloc}\n")
    file.write(f"Density Threshold : 10\n")
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


print(f"Elapsed time: {(time.time() - start_time)/60.} Minutes")

counter = Counter(reduction_factor)
pos_red = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pos_red.items()}

file_path = os.path.join(children_folder, f'reduction_factor10_{seed}.json')
with open(file_path, 'w') as json_file: json.dump(reduction_factor, json_file)

file_path = os.path.join(children_folder, f'numb_density10_{seed}.json')
with open(file_path, 'w') as json_file: json.dump(numb_density_at, json_file)

file_path = os.path.join(children_folder, f'position_vector10_{seed}')
with open(file_path, 'w') as json_file: json.dump(pos_red, json_file)

reduction_factor2 = list()
numb_density_at2  = list()
pos_red2 = dict()

N = threshold.shape[0] # number of lines

for cycle in range(max_cycles): # 100

    _from = N +1 - threshold2_rev[cycle]
    _to   = N+ 1 + threshold2[cycle] -1 # we sliced line[1:,:,:] so -1
    p_r = N + 1 - _from

    bfield    = magnetic_fields[_from:_to,cycle]
    distance = trajectory[_from:_to,cycle]
    numb_density = numb_densities[_from:_to,cycle]
    tupi = f"{x_init[cycle,0]},{x_init[cycle,1]},{x_init[cycle,2]}"
    print(len(bfield), p_r, threshold2[cycle], threshold2_rev[cycle])

    if bfield.shape[0] == 0:
        R = 1
        reduction_factor2.append(R)
        numb_density_at2.append(n_r)
        pos_red[tupi] = R
        continue

    if False:
        inter = (abs(np.roll(distance, 1) - distance) != 0) # removing pivot point
        distance = distance[inter]
        ds = np.abs(np.diff(distance, n=1))
        distance = distance[:-1]     # Remove the last element of distance
        numb_density  = numb_density[inter][:-1]
        bfield        = bfield[inter]
        adaptive_sigma = 3*ds/np.mean(ds)
        abfield = np.array([gaussian_filter1d(bfield, sigma=s, mode='nearest')[i] for i, s in enumerate(adaptive_sigma)])


    pocket, global_info = smooth_pocket_finder(bfield, cycle, plot=True) # this plots
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
        reduction_factor2.append(R)
        numb_density_at2.append(n_r)
        pos_red2[tupi] = R
        success = False 
        continue
    if success:
        if B_r / B_l < 1:
            R = 1 - np.sqrt(1 - B_r / B_l)
            reduction_factor2.append(R)
            numb_density_at2.append(n_r)
            pos_red2[tupi] = R
        else:
            R = 1
            reduction_factor2.append(R)
            numb_density_at2.append(n_r)
            pos_red2[tupi] = R

    print("Closest local maxima 'p':", closest_values)
    print("Bs: ", B_r, "ns: ", n_r)
    print("Bi: ", bfield[closest_values[0]], "Bj: ", bfield[closest_values[1]])

    if B_r/B_l < 1:
        print(" B_r/B_l =", B_r/B_l, "< 1 ") 
    else:
        print(" B_r/B_l =", B_r/B_l, "> 1 so CRs are not affected => R = 1") 

counter = Counter(reduction_factor2)

pos_red = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pos_red2.items()}

import datetime
import os, socket

hostname = socket.gethostname()

# Get current datetime
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(os.path.join(children_folder, f'PARAMETER_reduction100_{seed}'), 'w') as file:
    file.write(f"{filename}\n")
    file.write(f"File Created On: {current_datetime}\n")  # Add current datetime
    file.write(f"Hostname: {hostname}\n")
    file.write(f"Run ID/seed: {sys.argv[-1]}\n")
    file.write(f"{peak_den}\n")
    file.write(f"Cores Used: {os.cpu_count()}\n")
    file.write(f"Snap Time (Myr): {time_value}\n")
    file.write(f"Density Threshold : 100\n")
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
    file.write(f"Smallest Density (N/cm^3)  : {Density[np.argmax(Volume)]*gr_cm3_to_nuclei_cm3} \n")
    file.write(f"Biggest  Density (N/cm^3) : {Density[np.argmin(Volume)]*gr_cm3_to_nuclei_cm3}\n")
    file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")

print(f"Elapsed time: {(time.time() - start_time)/60.} Minutes")

counter = Counter(reduction_factor2)
pos_red = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in pos_red.items()}

file_path = os.path.join(children_folder, f'reduction_factor100_{seed}.json')
with open(file_path, 'w') as json_file: json.dump(reduction_factor2, json_file)

file_path = os.path.join(children_folder, f'numb_density100_{seed}.json')
with open(file_path, 'w') as json_file: json.dump(numb_density_at2, json_file)

file_path = os.path.join(children_folder, f'position_vector100_{seed}')
with open(file_path, 'w') as json_file: json.dump(pos_red2, json_file)

if True:
    try:
            
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
        plt.savefig(os.path.join(children_folder,f"FieldTopology{seed}.png"), bbox_inches='tight')

    except:
        print("Couldnt print B field structure")