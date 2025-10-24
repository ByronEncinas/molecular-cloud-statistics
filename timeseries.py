import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import csv, glob, os, sys, time
import h5py

from library import *

# python3 timeseries.py start_time/start_snap final_time/final_snap ideal/amb > R430TST.txt 2> R430TST_error.txt &

start_time = time.time()

FloatType = np.float64
IntType = np.int32


N               = 5_000
rloc            = 0.1
max_cycles      = 800
input_case      = 'ideal'
start_snap      = sys.argv[1]
num_file        = '430'
dense_cloud = 1.0e+1
threshold = 1.0e+1

file_xyz       = f'./{input_case}_cloud_trajectory.txt'

centerlst = []
densetlst = []
timestlst = []
snapstlst = []

with open(file_xyz, mode='r') as file:
    csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
    next(csv_reader)  # Skip the header row
    snap = []
    time_value = []
    for row in csv_reader:
        #print(row[0],row[1], np.log10(float(row[8])))
        if (row[0]) >= (start_snap):
            centerlst.append([float(row[2]),float(row[3]),float(row[4])])
            densetlst.append(float(row[8]))
            timestlst.append(float(row[1]))
            snapstlst.append(str(row[0]))
        else:
            break

centers    = np.array(centerlst)[::-1]
densities  = np.array(densetlst)[::-1]
timestlst  = np.array(timestlst)[::-1]
snapstlst  = np.array(snapstlst)[::-1]

if input_case == 'ideal':
    file_hdf5 = sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5'))
    subdirectory = 'ideal_mhd'
    for i, f in enumerate(file_hdf5):
        if f.split('.')[0][-3:] not in snapstlst:
            file_hdf5.remove(f)
elif input_case == 'amb':
    file_hdf5 = sorted(glob.glob('arepo_data/ambipolar_diffusion/*.hdf5'))
    subdirectory = 'ambipolar_diffusion'
    for i, f in enumerate(file_hdf5):
        if f.split('.')[0][-3:] not in snapstlst:
            file_hdf5.remove(f)

def reduction_density(reduction_data, density_data, bound = ''):

    reduction_data = np.array(reduction_data)
    density_data = np.array(density_data)
    def stats(n):
        sample_r = []
        for i in range(0, len(density_data)):
            if np.abs(np.log10(density_data[i]/n)) < 1/8:
                sample_r.append(reduction_data[i])
        sample_r.sort()
        if len(sample_r) == 0:
            return [np.nan, np.nan, np.nan]

        mean   = np.mean(sample_r)
        median = np.quantile(sample_r, .5)
        ten    = np.quantile(sample_r, .1)
        return [mean, median, ten]
        
    mask = reduction_data != 1
    reduction_data = reduction_data[mask]
    density_data = density_data[mask]
    fraction = (mask.shape[0] - np.sum(mask)) / mask.shape[0] # {R = 1}/{R}
    Npoints = len(reduction_data)
    n_min, n_max = np.log10(np.min(density_data)), np.log10(np.max(density_data))
    x_n = np.logspace(n_min, n_max, Npoints)
    mean_vec = np.zeros(Npoints)
    median_vec = np.zeros(Npoints)
    ten_vec = np.zeros(Npoints)
    for i in range(0, Npoints):
        s = stats(x_n[i])
        mean_vec[i] = s[0]
        median_vec[i] = s[1]
        ten_vec[i] = s[2]

    return x_n, mean_vec, median_vec, ten_vec, fraction
        
def pocket_finder(bfield, numb, p_r, plot=False):
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

    return (indexes, peaks), (index_global_max, upline)
        
def eval_reduction(field, numb, follow_index, threshold):

    R10      = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    #print("_, m = field.shape => ", _, m)

    flag = False
    filter_mask = np.ones(m)
    filter_mask[:] = True
    for i in range(m):

        mask10 = np.where(numb[:, i] > threshold)[0]

        if mask10.size > 0:
            start, end = mask10[0], mask10[-1]
            #print(mask10)
            #print(i, start, end)

            if start <= follow_index <= end:
                try:
                    numb10   = numb[start:end+1, i]
                    bfield10 = field[start:end+1, i]
                    p_r = follow_index - start
                    B_r = bfield10[p_r]
                    n_r = numb10[p_r]

                    #print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 
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

                #print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 
        else:
            print(f"\n[Info] No densities > {threshold} cm-3 found for column {i}. Using follow_index fallback.")
            if follow_index >= numb.shape[0]:
                raise ValueError(f"\nfollow_index {follow_index} is out of bounds for shape {numb.shape}")
            numb10   = np.array([numb[follow_index, i]])
            bfield10 = np.array([field[follow_index, i]])
            p_r = 0
            B_r = bfield10[p_r]
            n_r = numb10[p_r]

            #print("B_r ", B_r, "n_r ", n_r, "p_r", p_r) 
        
        # 0-2*l => x10-y10 so the new middle is l - x10  
        #print("p_r: ", p_r)
        if not (0 <= p_r < bfield10.shape[0]):
            raise IndexError(f"\np_r={p_r} is out of bounds for bfield10 of length {len(bfield10)}")

        #Min, max, and any zeros in numb: 0.0 710.1029394476656 True

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield10, numb10, p_r, plot=flag)
        index_pocket, field_pocket = pocket[0], pocket[1]
        flag = False

        p_i = np.searchsorted(index_pocket, p_r)
        # Flatten and count
        from collections import Counter
        counter = Counter(bfield10.ravel())  # ravel() flattens the array
        most_common_value, count = counter.most_common(1)[0]


        if count > 20:
            # if count > 20 we assume that there is a jump into low resolution in magnetic field
            # which happens at low densities, so we constrict field lines by this 
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)   
            flag = True
            print(f"Most common value: {most_common_value} (appears {count} times): R = ", R)
            filter_mask[i] = False # we erase this result so this is a mask to remove all field lines with errors
            continue      
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
     
    filter_mask = filter_mask.astype(bool)
    
    return np.array(R10), np.array(Numb100), np.array(B100), filter_mask  

def crs_path(x_init=np.array([0,0,0]),ncrit=threshold):
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init 
    
    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3
    k=0

    mask2 = dens > ncrit
    un_masked2 = np.logical_not(mask2) # 1 if finished

    x_rev = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x_rev, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol_rev = Volume[cells]

    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens_rev = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    k=0
    k_rev=0

    mask2_rev = dens > ncrit
    un_masked2_rev = np.logical_not(mask2_rev)

    while np.any(mask2) or np.any(mask2_rev): # 0 or 0 == 0 

        mask2_rev = dens_rev > ncrit
        un_masked2_rev = np.logical_not(mask2_rev)

        if np.sum(mask2_rev) > 0:

            x_rev_aux = x_rev[mask2_rev]
            x_rev_aux, bfield_aux_rev, dens_aux_rev, vol_rev = Heun_step(x_rev_aux, -1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
            dens_aux_rev = dens_aux_rev * gr_cm3_to_nuclei_cm3
            
            x_rev[mask2_rev] = x_rev_aux
            x_rev[un_masked2_rev] = 0
            dens_rev[mask2_rev] = dens_aux_rev
            dens_rev[un_masked2_rev] = 0
            line_rev[k_rev+1,mask2_rev,:] = x_rev_aux
            bfields_rev[k_rev+1,mask2_rev] = bfield_aux_rev
            densities_rev[k_rev+1,mask2_rev] = dens_aux_rev              

            k_rev += 1
        mask2 = dens > ncrit
        un_masked2 = np.logical_not(mask2) # 1 if finished
        
        if np.sum(mask2) > 0:
            x_aux = x[mask2]
            x_aux, bfield_aux, dens_aux, vol = Heun_step(x_aux, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
            dens_aux = dens_aux * gr_cm3_to_nuclei_cm3
            x[mask2] = x_aux
            x[un_masked2] = 0
            dens[mask2] = dens_aux
            dens[un_masked2] = 0

            line[k + 1, mask2, :]      = x_aux
            bfields[k + 1, mask2]      = bfield_aux
            densities[k + 1, mask2]    = dens_aux

            k += 1
        
        if (np.sum(mask2) == 0) and (np.sum(mask2_rev) == 0):
            print("There are no more points meeting the condition (e.g., density > 10cm-3).")
            break
        
        if k+1 > N or k_rev +1>N:
            print("Index is above allocated memory slot")
            break

        #print("counters (rev, fwd): ", k, k_rev)

    nz_i    = k + 1
    nz_irev = k_rev + 1
    
    print(f"get_lines => threshold index for {threshold}cm-3: ", nz_i, nz_irev)
    print(f"get_lines => original shapes ({2*N+1} to {nz_i + nz_irev - 1})")
    print(f"get_lines => p_r = {N+1} to p_r = {nz_irev} for array with shapes ...")

    radius_vectors = np.append(line_rev[:nz_irev,:,:][::-1, :, :], line[1:nz_i,:,:], axis=0)
    magnetic_fields = np.append(bfields_rev[:nz_irev,:][::-1, :], bfields[1:nz_i,:], axis=0)
    numb_densities = np.append(densities_rev[:nz_irev,:][::-1, :], densities[1:nz_i,:], axis=0)

    print("Radius vector shape:", radius_vectors.shape)

    m = magnetic_fields.shape[1]

    radius_vectors   *= 1.0#* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0#* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)

    return radius_vectors, magnetic_fields, numb_densities, nz_irev #p_r #, [threshold, threshold2, threshold_rev, threshold2_rev]
        
def line_of_sight(x_init=None, directions=fibonacci_sphere(), n_crit = threshold):
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
    #print(m0, l0)
    directions = np.tile(directions, (m0, 1))
    x_init = np.repeat(x_init, l0, axis=0)
    m = x_init.shape[0]
    l = directions.shape[0]
    #print(m, l)
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

        #print(k, dens[:2], dens_rev[:2])
        
        _, bfield, dens, vol = Heun_step(x, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)

        #_, bfield, dens, vol, ke, pressure = Heun_step(x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        
        #pressure *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens *= gr_cm3_to_nuclei_cm3
        
        vol[un_masked] = 0               # artifically make cell volume of finished lines equal to cero

        dx_vec = ((4 / 3) * vol / np.pi) ** (1 / 3)

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        x += dx_vec[:, np.newaxis] * directions

        line[k+1,:,:]    = x
        densities[k+1,:] = dens

        #_, bfield_rev, dens_rev, vol_rev, ke_rev, pressure_rev = Heun_step(x_rev, -1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        _, bfield_rev, dens_rev, vol_rev = Heun_step(x, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        #pressure_rev *= mass_unit / (length_unit * (time_unit ** 2)) 
        dens_rev *= gr_cm3_to_nuclei_cm3
        
        vol_rev[un_masked_rev] = 0 # unifinished lines will have cero volume for their corresponding cell

        dx_vec = ((4 / 3) * vol_rev / np.pi) ** (1 / 3)  # Increment step size

        threshold_rev += mask_rev.astype(int)  # Increment threshold count only for values still above 100

        x_rev -= dx_vec[:, np.newaxis] * directions

        line_rev[k+1,:,:]    = x_rev
        densities_rev[k+1,:] = dens_rev

        k_rev += 1
        k += 1
        #print(k, k_rev)
    
    threshold = threshold.astype(int)
    threshold_rev = threshold_rev.astype(int)

    max_th     = np.max(threshold) + 1
    max_th_rev = np.max(threshold_rev) + 1 

    radius_vectors = np.append(line_rev[:max_th_rev,:,:][::-1, :, :], line[1:max_th,:,:], axis=0)*pc_to_cm
    numb_densities = np.append(densities_rev[:max_th_rev,:][::-1, :], densities[1:max_th,:], axis=0)
    column_densities = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) # list of 'm' column densities
    
    #print(column_densities.shape, l0, m0)

    N_loss = np.zeros(m0)
    i = 0
    for x in range(0, l0*m0, l0): # l0 corresponds with how many directions we have
        N_loss[i] = np.mean(column_densities[x:x+l0])
        #print(i, x, x+l0, np.log10(N_loss[i]), column_densities[x:x+l0].shape)
        i += 1
    return radius_vectors, numb_densities, N_loss
        
def uniform_in_3d(no, rloc=1.0, ncrit=threshold): # modify
    def xyz_gen(size):
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U2 = np.random.uniform(low=0.0, high=1.0, size=size)
        U3 = np.random.uniform(low=0.0, high=1.0, size=size)
        r = rloc*np.cbrt(U1)
        theta = np.arccos(2*U2-1)
        phi = 2*np.pi*U3
        x,y,z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)

        rho_cartesian = np.array([[a,b,c] for a,b,c in zip(x,y,z)])
        rho_spherical = np.array([[a,b,c] for a,b,c in zip(r, theta, phi)])
        return rho_cartesian, rho_spherical

    from scipy.spatial import cKDTree
    from copy import deepcopy

    tree = cKDTree(Pos)
    valid_vectors = []
    rho_vector = np.zeros((no, 3))
    while len(valid_vectors) < no:
        aux_vector, _ = xyz_gen(no- len(valid_vectors)) # [[x,y,z], [x,y,z], ...] <= np array
        distances = np.linalg.norm(aux_vector, axis=1)
        inside_sphere = aux_vector[distances <= rloc]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > ncrit
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
        #print(ncrit, len(valid_vectors), no)
        if len(valid_vectors) == 0:
            ncrit /= 10
            flag = True
        del aux_vector, distances, inside_sphere, valid_mask, valid_points
        
    rho_vector = np.array(deepcopy(valid_vectors))

    if True:                   
        plt.scatter(rho_vector[:, 0], rho_vector[:,1], color = 'skyblue')
        plt.title('PDF $r = R\sqrt[3]{U(0,1)}$')
        plt.ylabel(r'PDF')
        plt.xlabel("$r$ (pc)")
        plt.grid()
        plt.tight_layout()
        plt.savefig('./images/xyz_pre_distro.png', dpi=300)
        plt.close()

    return rho_vector
        

timeseries = dict()

os.makedirs("series", exist_ok=True)

for each, filename in enumerate(file_hdf5):

    with open(file_xyz, mode='r') as file:

        csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
        next(csv_reader)  # Skip the header row
        print('File opened successfully')
        snap = []
        time_value = []
        for row in csv_reader:
            if num_file == str(row[0]):
                center = np.array([float(row[2]),float(row[3]),float(row[4])])
                snap =str(row[0])
                time_value = float(row[1])
                peak_den =  float(row[5])

    try:
        print(center)
    except NameError:
        import warnings
        warnings.warn("Center is not defined")
        continue

    snap = int(filename.split('.')[0][-3:])
    parent_folder = "series/" 
    data = h5py.File(filename, 'r')
    Boxsize = data['Header'].attrs['BoxSize']
    VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
    Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
    Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
    Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
    Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
    Bfield_grad = np.zeros((len(Pos), 9))
    Density_grad = np.zeros((len(Density), 3))
    Volume   = Mass/Density

    VoronoiPos-=center
    Pos       -=center
    for dim in range(3):  # Loop over x, y, z
        pos_from_center = Pos[:, dim]
        boundary_mask = pos_from_center > Boxsize / 2
        Pos[boundary_mask, dim] -= Boxsize
        VoronoiPos[boundary_mask, dim] -= Boxsize
    """
    for dim in range(3):
        pos_from_center = Pos[:, dim]

        too_high = pos_from_center > Boxsize / 2
        too_low  = pos_from_center < -Boxsize / 2

        Pos[too_high, dim] -= Boxsize
        Pos[too_low,  dim] += Boxsize
    """

    #x_input = np.vstack([uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud), np.array([0.0,0.0,0.0])])
    x_input = uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud)
    print(x_input.shape)

    # let's calculate field lines and lines of sight

    directions = fibonacci_sphere(20)

    radius_vectors, numb_densities, N_los                = line_of_sight(x_input, directions, threshold)
    radius_vectors, magnetic_fields, numb_densities, follow_index = crs_path(x_input, threshold)

    data.close()

    if np.any(numb_densities > threshold):
        r_u, n_rs, B_rs, filter_mask = eval_reduction(magnetic_fields, numb_densities, follow_index, threshold)
        r_l, _1, _2, _3= eval_reduction(magnetic_fields, numb_densities, follow_index, threshold*10)
        print("DEBUG numb_densities type:", type(numb_densities))
    else:
        print(f"Skipping evaluate_reduction: no densities above {threshold} cm⁻³")

    print(f"Elapsed Time: ", (time.time()-start_time)//60., " Minutes")


    print(type(filter_mask), filter_mask.dtype)
    print(filter_mask[:5])  
    print("removed lines",np.sum(np.logical_not(filter_mask)))  
    print("R_upper  = ", r_u.shape)
    print("R_lower = ", r_l.shape)
    print("B(r) = ", B_rs.shape)
    print("n(r) = ", n_rs.shape)

    u_input         = x_input[np.logical_not(filter_mask),:] 
    x_input         = x_input[filter_mask,:]
    radius_vectors  = radius_vectors[:, filter_mask, :]
    numb_densities  = numb_densities[:, filter_mask]
    N_los           = N_los[filter_mask]
    magnetic_fields = magnetic_fields[:, filter_mask]

    N_path   = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) *pc_to_cm

    print("R_upper  = ", r_u.shape)
    print("R_lower = ", r_l.shape)
    print("B(r) = ", B_rs.shape)
    print("n(r) = ", n_rs.shape)

    r_u, r_l, n_rs, B_rs = r_u[filter_mask], r_l[filter_mask], n_rs[filter_mask], B_rs[filter_mask]

    distance = np.linalg.norm(x_input, axis=1)*pc_to_cm

    holder = [snap, [r_u, r_l], [N_path, N_los], x_input, n_rs, B_rs]

    # save [time/snap, [Ru, Rl], [N_path, N_los], x_input, n_rs, B_rs]
    timeseries[(snap,)] = timeseries.get((snap,), holder * 0) + holder
#print(timeseries)


print("R_upper  = ", r_u.shape)
print("R_lower = ", r_l.shape)
print("B(r) = ", B_rs.shape)
print("n(r) = ", n_rs.shape)

r_bounds, r_means, r_var, r_skew, r_kur = [], [], [], [], []

r_means = np.mean(r_u)
r_var   = np.var(r_u)
r_skew  = np.array(skew(r_u))
r_kur  = np.array(kurtosis(r_u))

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plots = {
    'mean': {
        'y': r_means,
        'xlabel': '''$Time$ Myrs''',
        'ylabel': '$\mu$ (Mean)',
        'marker': 'o',
        'color': default_colors[0]
    },
    'std_dev': {
        'y': r_var,
        'xlabel': '''$Time$ Myrs''',
        'ylabel': '$\sigma$ (Std Dev)',
        'marker': 's',
        'color': default_colors[1]
    },
    'skewness': {
        'y': r_skew,
        'xlabel': '''$Time$ Myrs
        Sign of skewness points in the direction of the tail of the distro
        For $N(\mu, \sigma)$ the skewness $\gamma$ = 0''',
        'ylabel': '$\gamma$ (Skewness)',
        'marker': '^',
        'color': default_colors[2]
    },
    'kurtosis': {
        'y': r_kur,
        'xlabel': '''$Time$ Myrs
        Measures of whether or not a distribution is heavy-tailed or light-tailed
        relative to a $N(\mu, \sigma)$ the kurtosis $\kappa=3$''',
        'ylabel': '$\kappa$ (Kurtosis)',
        'marker': 'd',
        'color': default_colors[3]
    },
}

# For more information regarding the interpretation of skewness and kurtosis 
# https://www.statology.org/how-to-report-skewness-kurtosis/

# Loop through each plot and generate/save independently

x_snaps = [snapi[0] for snapi in timeseries.keys()]

for name, meta in plots.items():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_snaps, meta['y'], marker=meta['marker'], color=meta['color'])
    ax.set_ylabel(meta['ylabel'])
    ax.grid(True)
    if 'xlabel' in meta:
        ax.set_xlabel(meta['xlabel'])

    fig.suptitle(f'{name.capitalize()} Over Time (ideal)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./series/{name}_{input_case}_descriptors.png', dpi=300)
    plt.close()

m = radius_vectors.shape[1]

zoom = 2*rloc#np.mean(np.sqrt(x**2 + y**2 + z**2))-2

print(np.max(numb_densities))

if True:
           
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=np.min(magnetic_fields), vmax=np.max(magnetic_fields))
    cmap = cm.viridis

    ax = plt.figure().add_subplot(projection='3d')
    #radius_vectors /= pc_to_cm

    for k in range(m):

        x0=x_input[k, 0]
        y0=x_input[k, 1]
        z0=x_input[k, 2]

        ax.scatter(x0, y0, z0, marker="x",color="g",s=6)            
            
    ax.set_xlim(-zoom,zoom)
    ax.set_ylim(-zoom,zoom)
    ax.set_zlim(-zoom,zoom)
    ax.set_xlabel('x [Pc]')
    ax.set_ylabel('y [Pc]')
    ax.set_zlabel('z [Pc]')
    ax.set_title('Starting Points')

    # Add a colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Arbitrary Units')
    plt.savefig("./images/StartingPoints.png", bbox_inches='tight', dpi=300)

if True:
    try:
            
        from matplotlib import cm
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=np.min(magnetic_fields), vmax=np.max(magnetic_fields))
        cmap = cm.viridis

        ax = plt.figure().add_subplot(projection='3d')
        #radius_vectors /= pc_to_cm

        for k in range(m):
            # mask makes sure that start and ending point arent the zero vector
            mask = magnetic_fields[:, k] > 0

            x0=x_input[k, 0]
            y0=x_input[k, 1]
            z0=x_input[k, 2]

            x=radius_vectors[mask,k, 0]
            y=radius_vectors[mask,k, 1]
            z=radius_vectors[mask,k, 2]

            ax.scatter(x0, y0, z0, marker="x",color="black",s=3, alpha=0.2)            
            for l in range(len(x) - 1):
                color = cmap(norm(magnetic_fields[l, k]))
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=0.3)

            ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
            ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
                
        radius_to_origin = np.sqrt(x**2 + y**2 + z**2)
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
        cbar.set_label('Arbitrary Units')
        plt.savefig("./images/FieldTopology.png", bbox_inches='tight', dpi=300)

    except Exception as e:
        print(e)
        print("Couldnt print B field structure")

if __name__ == '__main__':
    print("Numb densities: ", numb_densities.shape)

    os.makedirs(parent_folder, exist_ok=True)
    h5_path = os.path.join(parent_folder, f"TimeBundle{snap}.h5")

    from filelock import FileLock
    """
    Time ──────────────────────────────▶

    Script 1:  ──[Acquire Lock]─────[Create/Append HDF5]─────[Release Lock]───

    Script 2:             ──[Wait for Lock]─────────────[Append HDF5]─────[Release Lock]───
    """
    lock = FileLock(h5_path + ".lock")

    with lock:  # acquire lock (waits if another process is holding it)

        if os.path.exists(h5_path):
            print("File exists:", h5_path)

            def append_to_dataset(f, name, new_data, axis):
                """
                Append new_data to dataset `name` inside open HDF5 file `f`.

                Parameters
                ----------
                f : h5py.File
                    An open HDF5 file (must be in "a" or "r+" mode).
                name : str
                    Dataset name.
                new_data : np.ndarray
                    Data to append. Must match dataset's shape in all but
                    the first dimension.
                """
                dset = f[name]
                dset_shape = dset.shape
                new_shape = list(dset_shape)
                if axis < 0:
                    axis += len(dset_shape)  # support negative axes

                for i, (dim_old, dim_new) in enumerate(zip(dset_shape, new_data.shape)):
                    if i != axis and dim_old != dim_new:
                        raise ValueError(
                            f"Dimension mismatch at axis {i}: dataset has {dim_old}, new_data has {dim_new}"
                        )

                # Resize along the append axis
                new_shape[axis] += new_data.shape[axis]
                dset.resize(new_shape)

                # Build a slicing object to assign new data
                slicer = [slice(None)] * len(dset_shape)
                slicer[axis] = slice(dset_shape[axis], new_shape[axis])
                dset[tuple(slicer)] = new_data     
                print("new shape", dset.shape)


            with h5py.File(h5_path, "a") as f:

                # --- Datasets ---
                append_to_dataset(f, "starting_point", x_input, 0)
                append_to_dataset(f, "number_densities", np.array(n_rs), 0)
                append_to_dataset(f, "magnetic_fields", np.array(B_rs), 0)
                append_to_dataset(f, "reduction_factor", np.array([r_u, r_l]), 1)  # shape (2, m)
                append_to_dataset(f, "column_path", N_path, 0)
                append_to_dataset(f, "column_los", N_los, 0)

                # --- Physical Parameters --- Dynamic Size makes appending difficult
                #print(numb_densities.shape)
                #append_to_dataset(f, "densities", numb_densities, 1)
                #print(magnetic_fields.shape)
                #append_to_dataset(f, "bfields", magnetic_fields, 1)
                #print(radius_vectors.shape)
                #append_to_dataset(f, "vectors", radius_vectors, 1)

            print(f"{h5_path.split('/')[-1]} Updated Successfully")
        else:
            print("File does not exist:", h5_path)

            # === Writing to HDF5 with metadata ===
            with h5py.File(h5_path, "w") as f:
                # Datasets, chunks=True
                f.create_dataset("starting_point", data=x_input, maxshape=(None, 3), chunks=True)
                f.create_dataset("number_densities", data=n_rs, maxshape=(None,)   , chunks=True)
                f.create_dataset("magnetic_fields", data=B_rs,maxshape=(None,)   , chunks=True)
                f.create_dataset("column_path", data=N_path, maxshape=(None,)   , chunks=True)
                f.create_dataset("reduction_factor", data=np.array([r_u, r_l]), maxshape=(None, None), chunks=True)
                f.create_dataset("column_los", data=N_los, maxshape=(None,), chunks=True)
                #f.create_dataset("bfields", data=magnetic_fields, maxshape=(None, None), chunks=True)
                #f.create_dataset("vectors", data=radius_vectors,maxshape=(None, None, 3), chunks=True)
                #f.create_dataset("densities", data=numb_densities, maxshape=(None, None), chunks=True)

                # === Metadata attributes ===
                f.attrs["cores_used"] = os.cpu_count()
                f.attrs["time"] = time_value
                f.attrs["rloc"] = rloc
                f.attrs["index"] = follow_index

                f.attrs["volume_range"] = [Volume[np.argmin(Volume)],Volume[np.argmax(Volume)]]
                f.attrs["density_range"] = [float(gr_cm3_to_nuclei_cm3 * Density[np.argmin(Volume)]) ,float(gr_cm3_to_nuclei_cm3 * Density[np.argmax(Volume)])]
                
            print(f"{h5_path.split('/')[-1]} Created Successfully")


    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    simulation_name = "Magnetic Pockets"
    runtime =(time.time()-start_time)//60


    pdf_file = "series/timeseries.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "Simulation Report")

    c.setFont("Helvetica", 12)
    lines = [
        f"Name:              {simulation_name}",
        f"Runtime:           {runtime:.2f} Minutes",
        f"Allocated Storage:  {2*(N+1)}",
        f"No. of Lines:       {max_cycles}",
        f"Assumptions:        {input_case}",
        f"Cloud Density:      {dense_cloud}",
        f"Threshold Density:  {threshold}",
        f"Filtered Lines:     {np.sum(np.logical_not(filter_mask))}",
        f"Final Lines:        {np.sum(filter_mask)}",
        f"Fraction Non-converging Lines:  {np.sum(np.logical_not(filter_mask))/np.sum(filter_mask)}",
    ]

    y = height - 120
    for line in lines:
        c.drawString(72, y, line)
        y -= 20

    c.save()
    print(f"Report saved to {pdf_file}")