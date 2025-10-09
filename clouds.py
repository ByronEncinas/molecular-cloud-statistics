import os, sys, time, glob, logging
import numpy as np, h5py, seaborn as sns
from scipy import stats
import yt; from yt import units as u
from library import *

# reference start time
start_time = time.time()

# Set yt logging to show only warnings and errors
yt.funcs.mylog.setLevel(logging.WARNING)

FloatType = np.float64
IntType = np.int32

try:
    input_case = str(sys.argv[1])
    how_many   = int(sys.argv[2])
except:
    input_case = 'amb'
    input_snap = '100'
    how_many   = 6

max_cycles = 100
rloc = 0.1

if input_case == 'ideal':
    file_list = sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5'))
elif input_case == 'amb':
    file_list = sorted(glob.glob('arepo_data/ambipolar_diffusion/*.hdf5'))

import os

centers = np.zeros((how_many,3))

for fileno, filename in enumerate(file_list):
    if input_snap not in filename:
        continue

    print(fileno, filename)

    data = h5py.File(filename, 'r')
    header_group = data['Header']
    time_value = header_group.attrs['Time']
    snap = filename.split('/')[-1].split('.')[0][-3:]
    parent_folder = "clouds/"+input_case
    children_folder = os.path.join(parent_folder, snap)
    os.makedirs(children_folder, exist_ok=True)
    Boxsize = data['Header'].attrs['BoxSize']
    Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
    Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
    
    #peak_density0 = Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3
    #center0 = Pos[np.argmax(Density)]

    xc = Pos[:, 0]
    yc = Pos[:, 1]
    zc = Pos[:, 2]

    region_radius = 1

    for each in range(how_many):            
        if each == 0:
            CloudCord = Pos[np.argmax(Density), :]
            PeakDensity = Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3
            with open(f"./clouds/{input_case}/{snap}/cloud.txt", "w") as file:
                file.write("snap,time_value,CloudCord_X,CloudCord_Y,CloudCord_Z,Peak_Density\n")
                file.write(f"{snap},{time_value},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")
            cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)
            Density[cloud_sphere] *= 0.0

            print(np.log10(PeakDensity))
        else:
            PeakDensity = Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3            
            CloudCord =   Pos[np.argmax(Density), :]
            cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)

            print(np.log10(PeakDensity))

            with open(f"./clouds/{input_case}/{snap}/cloud.txt", "a") as file:
                file.write(f"{snap},{time_value},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")
            Density[cloud_sphere] *= 0.0

        centers[each,:] = CloudCord.copy()

    ds = yt.load(filename)

    cm = np.mean(centers, axis=0)

    for each, cloud in enumerate(centers):
        sp = yt.SlicePlot(
            ds, 
            'y', 
            ('gas', 'density'), 
            center=[centers[each,0],centers[each,1],centers[each,2]],
            width = region_radius
        )
        sp.annotate_sphere(
            [cloud[0], cloud[1], cloud[2]],  # Center of the sphere
            radius=(5, "pc"),  # Radius of the sphere (in physical units, e.g., pc)
            circle_args={"color": "black", "linewidth": 2}  # Styling for the sphere
        )

        sp.annotate_sphere(
            [cloud[0], cloud[1], cloud[2]],  # Center of the sphere
            radius=(5, "pc"),  # Radius of the sphere (in physical units, e.g., pc)
            circle_args={"color": "black", "linewidth": 2}  # Styling for the sphere
        )

        sp.annotate_text(
            [cloud[0]+6, cloud[1]+6, cloud[2]],  # Slightly offset to avoid overlap
            text=f"{each}",  # Your custom label
            coord_system="data",  # Use data coordinates
            text_args={"color": "black", "fontsize": 12}
        )

        sp.annotate_timestamp(redshift=False)

        sp.save(f"./clouds/{input_case}/{snap}/{each}.png")
    data.close()


exit()
"""
# center at current cloud
Pos-=centers[each, :]

for dim in range(3):  # Loop over x, y, z
    pos_from_center = Pos[:, dim]

    # too far to the right
    too_high = pos_from_center > Boxsize / 2
    # too far to the left
    too_low  = pos_from_center < -Boxsize / 2

    Pos[too_high, dim] -= Boxsize
    Pos[too_low,  dim] += Boxsize
Pos+=centers[each, :]
"""
""" Define some useful functions"""

dense_cloud = 1.0e+2
threshold = 1.0e+2
N = 5_000

def eval_reduction(field, numb, follow_index, threshold):

    R10      = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    #print("_, m = field.shape => ", _, m)

    flag = False
       
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
     

    return R10, Numb100, B100

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

        # Create a mask for values that are 10^2 N/cm^3 above the threshold
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
            #volumes[k + 1, :]      = vol
            bfields[k + 1, mask2]      = bfield_aux
            densities[k + 1, mask2]    = dens_aux

            k += 1
            
        print(k, x_aux.shape, k_rev, x_rev_aux.shape) 
    
        if (np.sum(mask2) == 0) and (np.sum(mask2_rev) == 0):
            print("There are no more points meeting the condition (e.g., density > 10cm-3).")
            break
        
        if k+1 > N or k_rev +1>N:
            print("Index is above allocated memory slot")
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
        
        _, bfield, dens, vol = Heun_step(x, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)

        dens *= gr_cm3_to_nuclei_cm3
        
        vol[un_masked] = 0               # artifically make cell volume of finished lines equal to cero

        dx_vec = ((4 / 3) * vol / np.pi) ** (1 / 3)

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        x += dx_vec[:, np.newaxis] * directions

        line[k+1,:,:]    = x
        densities[k+1,:] = dens

        _, bfield_rev, dens_rev, vol_rev = Heun_step(x, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
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

def uniform_in_3d(no, rloc=1.0, ncrit=threshold): # modify
    def xyz_gen(size):
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U2 = np.random.uniform(low=0.0, high=1.0, size=size)
        U3 = np.random.uniform(low=0.0, high=1.0, size=size)
        r = rloc*np.cbrt(U1)
        theta = np.arccos(2*U2-1)
        phi = 2*np.pi*U3
        x,y,z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
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

""" From this point on, we can loop through the clouds calculating the column densities"""

column_los_cloud  = np.zeros_like((max_cycles, how_many))
column_path_cloud = np.zeros_like((max_cycles, how_many))
density_cloud = np.zeros_like((max_cycles, how_many))
ratio_column_path_to_los = np.zeros_like((max_cycles, how_many))

for list in file_list:
    for fileno, filename in enumerate(list[::-1]):
        print(f"{filename}")

        snap = filename.split('/')[-1].split('.')[0][-3:]
        data = h5py.File(filename, 'r')
        header_group = data['Header']
        time = header_group.attrs['Time']
        children_folder = os.path.join("clouds/"+input_case, snap)
        print(children_folder)
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

        # center at current cloud
        Pos-=centers[each, :]
        VoronoiPos-=centers[each, :]

        for dim in range(3):  # Loop over x, y, z
            pos_from_center = Pos[:, dim]

            too_high = pos_from_center > Boxsize / 2
            too_low  = pos_from_center < -Boxsize / 2

            Pos[too_high, dim] -= Boxsize
            Pos[too_low,  dim] += Boxsize

        #x_input = np.vstack([uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud), np.array([0.0,0.0,0.0])])
        x_input = uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud)
        directions = fibonacci_sphere(10)

        radius_vector, numb_densities, N_los = line_of_sight(x_input, directions, threshold)
        radius_vector, magnetic_fields, numb_densities, follow_index = crs_path(x_input, threshold)

        print(f"Elapsed Time: ", (time.time()-start_time)//60., " Minutes")

        if np.any(numb_densities > threshold):
            r_u, n_rs, B_rs = eval_reduction(magnetic_fields, numb_densities, follow_index, threshold)
            r_l, _, _ = eval_reduction(magnetic_fields, numb_densities, follow_index, threshold/10)
            print("DEBUG eval_reduction __func__")
        else:
            print(f"Skipping evaluate_reduction: no densities above {threshold} cm⁻³")

        print(f"Elapsed Time: ", (time.time()-start_time)//60., " Minutes")

        distance = np.linalg.norm(x_input, axis=1)*pc_to_cm
        # cr path column densities
        N_path   = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vector, axis=0), axis=2), axis=0) *pc_to_cm

        # free space
        Pos+=centers[each, :]
        VoronoiPos+=centers[each, :]

        print((time.time()-start_time)//60, " Minutes")

        #distance = np.linalg.norm(x_input, axis=1)*pc_to_cm

        # Isolate data with the corresponding cloud 
        column_los_cloud[:,each]   = N_los
        column_path_cloud[:,each]  = N_path
        density_cloud[:,each]      = n_rs
        ratio_column_path_to_los[:,each]   = N_path / N_los

        # Ionization is not as important as determining which column density remains statistically higher

""" Here we will plot the Ration Npath/Nlos and its 95% confidence band on a mosaic containing 6 different molecular clouds"""

fig, axs = plt.subplots(2, 3, figsize=(15, 8),gridspec_kw={'wspace': 0, 'hspace': 0},sharex=True, sharey='row')

column_labels = [r'$\Sigma = 1\,\mathrm{g\,cm^{-2}}$', 
                 r'$\Sigma = 100\,\mathrm{g\,cm^{-2}}$', 
                 r'$\Sigma = 1000\,\mathrm{g\,cm^{-2}}$']

E = np.logspace(4, 11, 300)
logE = np.log10(E)

# Dummy Y data
def dummy_curve(scale=1, slope=-2):
    return np.log10(scale * E**slope + 1e-30)

# Line styles
line_styles_top = {
    'C1': {'color': 'orange', 'linewidth': 2},
    'C1': {'color': 'orange', 'linewidth': 2},
    'C2': {'color': 'blue', 'linewidth': 2},
    'C3': {'color': 'green', 'linewidth': 2}
}

line_styles_bottom = {
    'C4': {'color': 'red', 'linewidth': 2},
    'C5': {'color': 'orange', 'linewidth': 2, 'linestyle': '--'},
    'C6': {'color': 'gold', 'linewidth': 2, 'linestyle': ':'}
}

# Loop over columns
for i in range(3):
    # Top row: photon spectra
    ax_top = axs[0, i]
    for label, style in line_styles_top.items():
        ax_top.plot(density_cloud[:, i], ratio_column_path_to_los[:, each+3], label=label, **style)
    ax_top.set_title(column_labels[i], fontsize=14)
    if i == 0:
        ax_top.set_ylabel(r'$$\log_{10}[\frac{N_{path}}{N_{los}}$$', fontsize=12)
    ax_top.grid(True)

    # Bottom row: electron/positron spectra
    ax_bottom = axs[1, i]
    for label, style in line_styles_bottom.items():
        ax_bottom.plot(density_cloud[:, i], ratio_column_path_to_los[:, each+3], label=label, **style)
    if i == 0:
        ax_bottom.set_ylabel(r'$\log_{10}[\frac{N_{path}}{N_{los}}$', fontsize=12)
    ax_bottom.set_xlabel(r'$\log_{10}[n_g/\mathrm{cm^{-3}}]$', fontsize=12)
    ax_bottom.grid(True)

# Legend handling (only one legend for each row)
axs[0, 2].legend(loc='upper right', fontsize=9)
axs[1, 2].legend(loc='lower left', fontsize=9)

# Adjust layout
plt.tight_layout()
plt.savefig(os.path.join(children_folder, f"{input_case}_{snap}.png"))
plt.close(fig)

""" Here we will plot slice in form of kde_curves or hst_curves that map this ratio for each plot"""


if False: # Ratio N_path/N_los
    fig, ax = plt.subplots()
    
    ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
    ax.set_ylabel(r'$ratio$', fontsize=16)
    ax.set_yscale('log')
    #ax.set_xscale('log')

    ax.scatter(s, N_path/N_los, marker ='v',color="black", s=5, label=r'$N_{\rm path}/N_{\rm los}$', alpha=0.3)
    ax.legend(loc="upper left", fontsize=12)    
    ax.tick_params(axis='both')
    ax.grid(True, which='both', alpha=0.3)
    
    plt.title(r"Ratio $N_{path}/N_{los}$", fontsize=16)
    fig.tight_layout()
    plt.savefig('./images/columns/column_ratio.png', dpi=300)
    plt.close()

if False:

    fig, (ax_l, ax_h) = plt.subplots(1, 2, figsize=(10, 5),gridspec_kw={'wspace': 0, 'hspace': 0}, sharey=True)

    ax_l.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
    ax_l.set_ylabel(r'$log_{10}(\zeta /\rm{cm}^{-2})$', fontsize=16)
    ax_l.set_xscale('log')
    ax_l.scatter(n_rs, log_ionization_los_l, marker='o', color="#8E2BAF", s=5, alpha=0.3, label=r'$\zeta_{\rm los}$')
    ax_l.scatter(n_rs, log_ionization_path_l, marker='v', color="#148A02", s=5, alpha=0.3, label=r'$\zeta_{\rm path}$')
    ax_l.grid(True, which='both', alpha=0.3)
    ax_l.legend(fontsize=16)
    ax_l.set_ylim(-19.5, -15.5)
    ax_l.set_title("Model $\mathcal{L}$", fontsize=16)

    ax_h.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
    ax_h.set_xscale('log')
    ax_h.scatter(n_rs, log_ionization_los_h, marker='o', color="#8E2BAF", s=5, alpha=0.3, label=r'$\zeta_{\rm los}$')
    ax_h.scatter(n_rs, log_ionization_path_h, marker='v', color="#148A02", s=5, alpha=0.3, label=r'$\zeta_{\rm path}$')
    ax_h.grid(True, which='both', alpha=0.3)
    ax_h.legend(fontsize=16)
    ax_h.set_ylim(-19.5, -15.5)
    ax_h.set_title("Model $\mathcal{H}$", fontsize=16)
    ax_h.tick_params(labelleft=False)

    fig.suptitle(r"Ionization Rate ($\zeta$) vs. Density ($n_g$)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('./images/columns/zeta_density_combined.png', dpi=300)
    plt.close()
