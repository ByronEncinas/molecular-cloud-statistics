from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from library import *
import csv, glob, os, sys, time
import h5py

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
    N               = 5_000
    rloc            = 0.5
    max_cycles      = 200
    case            = 'ideal'
    num_file        = '430'
    seed            = 12345 
    sys.argv.append(seed)

ncrit = 1.0e+2
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
file_list = glob.glob(f"arepo_data/{subdirectory}/*.hdf5")

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
filename = None

for f in file_list:
    if num_file in f:
        filename = f
if filename is None:
    raise FileNotFoundError

os.makedirs("stats", exist_ok=True)
parent_folder = "stats/"+ case 
children_folder = os.path.join(parent_folder, snap)
os.makedirs(children_folder, exist_ok=True)

data = h5py.File(filename, 'r')
header_group = data['Header']
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

for dim in range(3):  # Loop over x, y, z
    pos_from_center = Pos[:, dim]
    boundary_mask = pos_from_center > Boxsize / 2
    Pos[boundary_mask, dim] -= Boxsize
    VoronoiPos[boundary_mask, dim] -= Boxsize

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
print("Allocation Number: ", 2*N)

tmp_file = os.path.join(children_folder, "tmp.csv")
with open(tmp_file, "w"):
    print('Temp file created successfully')
    pass  # Nothing to write, just clears the file

def pocket_finder(bfield, numb, p_r, plot=False):
    #pocket_finder(bfield, p_r, B_r, img=i, plot=False)
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    try:
        idx = index_global_max[0]
    except:
        idx = index_global_max
    upline == bfield[idx]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001


    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]

    if plot:
        # Find threshold crossing points for 100 cm^-3
        mask = np.log10(numb) < 2  # log10(100) = 2
        slicebelow = mask[:p_r]
        sliceabove = mask[p_r:]
        peaks = np.array(peaks)
        indexes = np.array(indexes)

        try:
            above100 = np.where(sliceabove)[0][0] + p_r
        except IndexError:
            above100 = None

        try:
            below100 = np.where(slicebelow)[0][-1]
        except IndexError:
            below100 = None

        # Create a mosaic layout with two subplots: one for 'numb', one for 'bfield'
        fig, axs_dict = plt.subplot_mosaic([['numb', 'bfield']], figsize=(12, 5))
        axs_numb = axs_dict['numb']
        axs_bfield = axs_dict['bfield']

        def plot_field(axs, data, label):

            axs.plot(data, label=label)
            if below100 is not None:
                axs.vlines(below100, data[below100]*(1 - 0.1), data[below100]*(1 + 0.1),
                        color='black', label='th 100cm⁻³ (left)')
            if above100 is not None:
                axs.vlines(above100, data[above100]*(1 - 0.1), data[above100]*(1 + 0.1),
                        color='black', label='th 100cm⁻³ (right)')
            if peaks is not None:
                axs.plot(indexes, data[indexes], "x", color="green", label="all peaks")
                axs.plot(indexes, data[indexes], ":", color="green")

            if idx is not None and upline is not None:
                axs.plot(idx, np.max(data), "x", color="black", label="index_global_max")

            axs.axhline(np.min(data), linestyle="--", color="gray", label="baseline")

            axs.set_xlabel("Index")
            axs.set_ylabel(label)
            axs.set_title(f"{label} Shape")
            axs.legend()
            axs.grid(True)

        # Plot both subplots
        plot_field(axs_numb, numb, "Density")
        plot_field(axs_bfield, bfield, "Magnetic Field")

        plt.tight_layout()
        plt.savefig('./mosaic.png')
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def evaluate_reduction(field, numb, follow_index):

    # field and numb are retrieved 
    if not np.any(numb > 100):
        print("No densities > 100 cm⁻³ found. Skipping evaluation.")
        return None, None, None, None

    R10      = []
    R100     = []
    Numb100  = []
    B100     = []

    _, m = field.shape
       
    for i in range(m):

        #_100 = np.where(numb[:, i] > 1.0e+2)[0]
        #_10  = np.where(numb[:, i] > 1.0e+1)[0]
        # must be k + k_rev + 1 in size
        print("DEBUG SHAPE CHECK:", numb[:, i].shape) 
               
        # this must contain list of indexes of positions with densities > 10cm-3
        # this should be empty for at least one
        mask10 = np.where(numb[:, i] > 1.0e+1)[0]

        if mask10.size > 0:
            start, end = mask10[0], mask10[-1]
            #print(mask10)
            print(start, end)

            if start <= follow_index <= end:
                try:
                    numb10   = numb[start:end+1, i]
                    bfield10 = field[start:end+1, i]
                    p_r = follow_index - start
                    B_r = bfield10[p_r]
                    n_r = numb10[p_r]

                    print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 
                except IndexError:
                    raise ValueError(f"\nTrying to slice beyond bounds for column {i}. "
                                    f"start={start}, end={end}, shape={numb.shape}")
            else:
                print(f"\n[Info] follow_index {follow_index} outside threshold interval for column {i}.")
                if follow_index >= numb.shape[0]:
                    raise ValueError(f"follow_index {follow_index} is out of bounds for shape {numb.shape}")
                numb10   = np.array([numb[follow_index, i]])
                bfield10 = np.array([field[follow_index, i]])
                p_r = 0
                B_r = bfield10[p_r]
                n_r = numb10[p_r]

                print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 
        else:
            print(f"\n[Info] No densities > 10 cm⁻³ found for column {i}. Using follow_index fallback.")
            if follow_index >= numb.shape[0]:
                raise ValueError(f"\nfollow_index {follow_index} is out of bounds for shape {numb.shape}")
            numb10   = np.array([numb[follow_index, i]])
            bfield10 = np.array([field[follow_index, i]])
            p_r = 0
            B_r = bfield10[p_r]
            n_r = numb10[p_r]

            print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 

        
        print("shapes: n, b: ", numb10.shape, bfield10.shape)
        
        # 0-2*l => x10-y10 so the new middle is l - x10  
        print("p_r: ", p_r)
        if not (0 <= p_r < bfield10.shape[0]):
            raise IndexError(f"\np_r={p_r} is out of bounds for bfield10 of length {len(bfield10)}")

        #Min, max, and any zeros in numb: 0.0 710.1029394476656 True

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield10, numb10, p_r, plot=True)
        index_pocket, field_pocket = pocket[0], pocket[1]

        p_i = np.searchsorted(index_pocket, p_r)
        
        # are there local maxima around our point? 
        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            B_h = max([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            # YES! 
            success = True  
        except:
            # NO :c
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)
            else:
                # NO!
                R = 1.
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)

        del closest_values, success, B_l, B_h, R, p_r
        
        # this must be an array 
        mask100 = np.where(numb[:, i] > 1.0e+2)[0]
        
        if mask100.size > 0:
            # both integers
            start, end = mask100[0], mask100[-1]

            if start <= follow_index <= end:
                try:
                    numb100   = numb[start:end+1, i]
                    bfield100 = field[start:end+1, i]
                    p_r = follow_index - start
                    B_r = bfield100[p_r]
                    n_r = numb100[p_r]

                    print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 

                except IndexError:
                    raise ValueError(f"\nTrying to slice beyond bounds for column {i}. "
                                    f"\nstart={start}, end={end}, shape={numb.shape}")
            else:
                print(f"\n[Info] follow_index {follow_index} outside threshold interval for column {i}.")
                if follow_index >= numb.shape[0]:
                    raise ValueError(f"follow_index {follow_index} is out of bounds for shape {numb.shape}")
                numb100   = np.array([numb[follow_index, i]])
                bfield100 = np.array([field[follow_index, i]])
                p_r = 0
                B_r = bfield10[p_r]
                n_r = numb10[p_r]

                print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 

        else:
            print(f"\n[Info] No densities > 100 cm⁻³ found for column {i}. Using follow_index fallback.")
            if follow_index >= numb.shape[0]:
                raise ValueError(f"\nfollow_index {follow_index} is out of bounds for shape {numb.shape}")
            numb100   = np.array([numb[follow_index, i]])
            bfield100 = np.array([field[follow_index, i]])
            p_r = 0
            B_r = bfield10[p_r]
            n_r = numb10[p_r]

            print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 

        print("shapes: n, b: ", numb100.shape, bfield100.shape)
        
        # 0-2*l => x10-y10 so the new middle is l - x10  
        print("p_r: ", p_r)
        if not (0 <= p_r < bfield100.shape[0]):
            raise IndexError(f"p_r={p_r} is out of bounds for bfield10 of length {len(bfield100)}")

        #Min, max, and any zeros in numb: 0.0 710.1029394476656 True

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield100, numb100, p_r, plot=False)
        index_pocket, field_pocket = pocket[0], pocket[1]

        p_i = np.searchsorted(index_pocket, p_r)
        
        # are there local maxima around our point? 
        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield100[closest_values[0]], bfield100[closest_values[1]]])
            B_h = max([bfield100[closest_values[0]], bfield100[closest_values[1]]])
            # YES! 
            success = True  
        except:
            # NO :c
            R = 1.
            R100.append(R)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R100.append(R)
            else:
                # NO!
                R = 1.
                R100.append(R)

    return R10, R100, Numb100, B100

def get_along_lines(x_init=np.array([0,0,0]),ncrit=ncrit):
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))
    #volumes   = np.zeros((N+1,m))
    #threshold = np.zeros((m,)).astype(int) # one value for each
    #threshold2 = np.zeros((m,)).astype(int) # one value for each

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
    #volumes_rev   = np.zeros((N+1,m))
    #threshold_rev = np.zeros((m,)).astype(int) # one value for each
    #threshold2_rev = np.zeros((m,)).astype(int) # one value for each

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init 
    
    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3
    k=0

    #mask  = dens > 1.0e+1# 1 if not finished
    #un_masked = np.logical_not(mask) # 1 if finished

    mask2 = dens > ncrit
    un_masked2 = np.logical_not(mask2) # 1 if finished

    x_rev = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x_rev, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol_rev = Volume[cells]

    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens_rev = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    k=0
    k_rev=0

    #mask_rev = dens > 1.0e+1
    #un_masked_rev = np.logical_not(mask_rev)
    mask2_rev = dens > ncrit
    un_masked2_rev = np.logical_not(mask2_rev)

    if m < 100:
        buffer = 50 # 131072 # 128 KB
    else:
        buffer = 1048576 # 1 MB

    with open(tmp_file, "a", buffering=buffer) as f: 
        combined = np.concatenate([line[k, :, :].flatten(), line_rev[k, :, :].flatten(), densities[k, mask2].flatten(), densities_rev[k, mask2].flatten()]) #x.flatten(), x_rev.flatten(), 
        csv_line = ','.join(map(str, combined))
        f.write(csv_line + '\n')

        while np.any(mask2) or np.any(mask2_rev): # 0 or 0 == 0 

            #mask_rev = dens > 1.0e+1
            #un_masked_rev = np.logical_not(mask_rev)
            mask2_rev = dens_rev > ncrit
            un_masked2_rev = np.logical_not(mask2_rev)

            if np.sum(mask2_rev) > 0:

                #aux =  x[un_masked_rev]
                #aux2 = x[un_masked2_rev]
                x_rev_aux = x_rev[mask2_rev]

                x_rev_aux, bfield_aux_rev, dens_aux_rev, vol_rev = Heun_step(x_rev_aux, -1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
                dens_aux_rev = dens_aux_rev * gr_cm3_to_nuclei_cm3
                
                x_rev[mask2_rev] = x_rev_aux
                x_rev[un_masked2_rev] = 0
                dens_rev[mask2_rev] = dens_aux_rev
                dens_rev[un_masked2_rev] = 0

                #print(x_rev_aux.shape, x_rev.shape) 
                #threshold_rev += mask_rev.astype(int)
                #threshold2_rev += mask2_rev.astype(int)
                #threshold2[mask2_rev] += 1

                #x[un_masked2_rev] = 0.0

                line_rev[k_rev+1,mask2_rev,:] = x_rev_aux
                #volumes_rev[k_rev+1,:] = vol_rev
                bfields_rev[k_rev+1,mask2_rev] = bfield_aux_rev
                densities_rev[k_rev+1,mask2_rev] = dens_aux_rev              

                k_rev += 1

            # Create a mask for values that are 10^2 N/cm^3 above the threshold
            #mask  = dens > 1.0e+1 # 1 if not finished
            #un_masked = np.logical_not(mask) # 1 if finished
            mask2 = dens > ncrit
            un_masked2 = np.logical_not(mask2) # 1 if finished
            
            if np.sum(mask2) > 0:
                #aux  =x[un_masked]
                #aux2 =x[un_masked2]

                x_aux = x[mask2]
                x_aux, bfield_aux, dens_aux, vol = Heun_step(x_aux, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
                dens_aux = dens_aux * gr_cm3_to_nuclei_cm3
                x[mask2] = x_aux
                x[un_masked2] = 0
                dens[mask2] = dens_aux
                dens[un_masked2] = 0
                #threshold  += mask.astype(int)  # Increment threshold count only for values still above 100
                #threshold2[mask2] += mask2.astype(int)  # Increment threshold count only for values still above 100
                #threshold2[mask2] += 1
                #x[un_masked2] = 0.0

                line[k + 1, mask2, :]      = x_aux
                #volumes[k + 1, :]      = vol
                bfields[k + 1, mask2]      = bfield_aux
                densities[k + 1, mask2]    = dens_aux

                k += 1
                
            combined = np.concatenate([line[k, :, :].flatten(), line_rev[k_rev, :, :].flatten(), densities[k, :].flatten(), densities_rev[k_rev, :].flatten()]) #x.flatten(), x_rev.flatten(), 
            csv_line = ','.join(map(str, combined))
            f.write(csv_line + '\n')

            print(max(np.max(np.linalg.norm(line[k, :, :], axis=1)),np.max(np.linalg.norm(line_rev[k_rev, :, :], axis=1))))

            #print(k, x_aux.shape, k_rev, x_rev_aux.shape) 

        
            if (np.sum(mask2) == 0) and (np.sum(mask2_rev) == 0):
                print("There are no more points meeting the condition (e.g., density > 10cm-3).")
                break

    print("counters (rev, fwd): ", k, k_rev)
    #threshold = threshold.astype(int)

    #updated_mask = np.logical_not(np.logical_and(mask, mask_rev))

    # Apply updated_mask to the second axis of (N+1, m, 3) or (N+1, m) arrays
    #line = line[:, updated_mask, :]  # Mask applied to the second dimension (m)
    #volumes = volumes[:, updated_mask]  # Assuming volumes has shape (m,)
    #bfields = bfields[:, updated_mask]  # Mask applied to second dimension (m)
    #densities = densities[:, updated_mask]  # Mask applied to second dimension (m)

    # Apply to the reverse arrays in the same way
    #line_rev = line_rev[:, updated_mask, :]
    #volumes_rev = volumes_rev[:, updated_mask]
    #bfields_rev = bfields_rev[:, updated_mask]
    #densities_rev = densities_rev[:, updated_mask]
    
    nz_i    = k + 1
    nz_irev = k_rev + 1
    
    print(f"get_lines => threshold index for 10cm-3: ", nz_i, nz_irev)
    print(f"get_lines => original shapes ({2*N+1} to {nz_i + nz_irev - 1})")
    print(f"get_lines => p_r = {N+1} to p_r = {nz_irev} for array with shapes ...")

    radius_vector = np.append(line_rev[:nz_irev,:,:][::-1, :, :], line[1:nz_i,:,:], axis=0)
    magnetic_fields = np.append(bfields_rev[:nz_irev,:][::-1, :], bfields[1:nz_i,:], axis=0)
    numb_densities = np.append(densities_rev[:nz_irev,:][::-1, :], densities[1:nz_i,:], axis=0)

    #N = magnetic_fields.shape[0]

    print("Radius vector shape:", radius_vector.shape)

    m = magnetic_fields.shape[1]

    radius_vector   *= 1.0#* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0#* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)

    return radius_vector, magnetic_fields, numb_densities, nz_irev #p_r #, [threshold, threshold2, threshold_rev, threshold2_rev]

def uniform_in_3d(no, rloc=1.0, ncrit=1.0e+2): # modify
    def xyz_gen(size):
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U2 = np.random.uniform(low=0.0, high=1.0, size=size)
        U3 = np.random.uniform(low=0.0, high=1.0, size=size)
        r = rloc*np.cbrt(U1)
        theta = np.arccos(2*U2-1)
        phi = 2*np.pi*U3
        x,y,z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
        if False:                   
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
        return np.array([[a,b,c] for a,b,c in zip(x,y,z)])

    from scipy.spatial import cKDTree
    from copy import deepcopy

    tree = cKDTree(Pos)
    valid_vectors = []
    rho_vector = np.zeros((no, 3))
    while len(valid_vectors) < no:
        aux_vector = xyz_gen(no- len(valid_vectors)) # [[x,y,z], [x,y,z], ...] <= np array
        distances = np.linalg.norm(aux_vector, axis=1)
        inside_sphere = aux_vector[distances <= rloc]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > ncrit
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
        del aux_vector, distances, inside_sphere, valid_mask, valid_points
    rho_vector = np.array(deepcopy(valid_vectors))
    return rho_vector

x_input = np.vstack([uniform_in_3d(max_cycles, rloc, ncrit=10e+4), np.array([0.0,0.0,0.0])])

# x_input provides with the corresponding values to r_100 and r_10
#x_input = uniform_in_3d(max_cycles, rloc, ncrit=1.0e+2)

radius_vector, magnetic_fields, numb_densities, follow_index = get_along_lines(x_input, ncrit)

print(f"Elapsed Time: ", (time.time()-start_time)//60., " Minutes")

if np.any(numb_densities > 1.0e2):
    r_10, r_100, n_rs, B_rs = evaluate_reduction(magnetic_fields, numb_densities, follow_index)
    print("DEBUG numb_densities type:", type(numb_densities))
else:
    print("Skipping evaluate_reduction: no densities above 100 cm⁻³")

print(f"Elapsed Time: ", (time.time()-start_time)//60., " Minutes")

# cr path column densities
c_rs   = np.cumsum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vector, axis=0), axis=2), axis=0) 

print(f"Elapsed Time: ", (time.time()-start_time)//60., " Minutes")

print("R10  = ", r_10)
print("R100 = ", r_100)
print("B(r) = ", B_rs)
print("n(r) = ", n_rs)

os.makedirs(children_folder, exist_ok=True)
h5_path = os.path.join(children_folder, f"DataBundle{seed}.h5")

m = magnetic_fields.shape[1]
# === Writing to HDF5 with metadata ===
with h5py.File(h5_path, "w") as f:
    # Datasets
    #f.create_dataset("positions", data=radius_vector)
    #f.create_dataset("densities", data=numb_densities)
    #f.create_dataset("magnetic_fields", data=magnetic_fields)
    f.create_dataset("starting_point", data=x_input)
    f.create_dataset("number_densities", data=n_rs)
    f.create_dataset("magnetic_fields", data=B_rs)
    f.create_dataset("column_densities", data=c_rs)
    f.create_dataset("reduction_factor", data=np.array([r_10, r_100]))

    # === Metadata attributes ===
    #f.attrs["seed"] = seed
    f.attrs["cores_used"] = os.cpu_count()
    f.attrs["pre_allocation_number"] = 2 * N
    f.attrs["rloc"] = rloc
    f.attrs["max_cycles"] = m
    f.attrs["center"] = Center
    f.attrs["volume_range"] = [Volume[np.argmin(Volume)],Volume[np.argmax(Volume)]]
    f.attrs["density_range"] = [float(gr_cm3_to_nuclei_cm3 * Density[np.argmin(Volume)]) ,float(gr_cm3_to_nuclei_cm3 * Density[np.argmax(Volume)])]
    
    try:
        pass
        os.remove(tmp_file)
    except:
        print(f"Temp file not found tmp.csv")
        raise FileNotFoundError

print(f"{h5_path.split('/')[-1]} Created Successfully")

# free space
del Mass, Volume, Density_grad, Density
del Bfield, Bfield_grad, Pos, VoronoiPos
del data

distance = np.linalg.norm(x_input, axis=1)*pc_to_cm

if True:
    fig, ax = plt.subplots()
    
    # Axis labels
    ax.set_xlabel(r'''Distance $r$ (cm)
                
    Column density along line of sight as a function of distance 
    away from core
            $N_{PATH} = \int_0^s n_g(s')ds'$   
            ''')
    
    ax.set_ylabel(r'$N_{\rm LOS}$ (cm$^{-2}$)', fontsize=12)
    
    # Log scales
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # Scatter plot with label for legend
    ax.scatter(distance, c_rs, color='dodgerblue', s=2.0, label=r'$\langle N_{\rm LOS} \rangle$')
    
    # Ticks and grid
    ax.tick_params(axis='both')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Title and legend
    plt.title(r"$N_{\rm LOS} \propto r$", fontsize=13)
    ax.legend()
    
    # Layout and save
    fig.tight_layout()
    plt.savefig('./avg_column_path.png')
    plt.close()

if True:
    try:
            
        from matplotlib import cm
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=np.min(magnetic_fields), vmax=np.max(magnetic_fields))
        cmap = cm.viridis

        ax = plt.figure().add_subplot(projection='3d')
        radius_vector /= 3.086e+18

        for k in range(m):
            # mask makes sure that start and ending point arent the zero vector
            numb_densities[:, k]
            mask = numb_densities[:, k] > 0

            x=radius_vector[mask, k, 0]
            y=radius_vector[mask, k, 1]
            z=radius_vector[mask, k, 2]
            
            for l in range(len(x) - 1):
                color = cmap(norm(magnetic_fields[l, k]))
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=0.3)

            ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
            ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
                
        radius_to_origin = np.sqrt(x**2 + y**2 + z**2)
        zoom = (np.max(radius_to_origin) + rloc)/2.0
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