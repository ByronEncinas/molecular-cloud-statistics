from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import csv, glob, os, sys, time, h5py, gc
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.spatial import cKDTree
import warnings
from src.library import *
from mpi4py import MPI          # Move to TOP of __main__
import pickle
import asyncio

start_time = time.time()

@timing
def uniform_in_3d_tree_dependent(tree, no, rloc=1.0, n_crit=1.0e+2):

    def xyz_gen(size, _r):          # ← accepts current radius as argument
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U2 = np.random.uniform(low=0.0, high=1.0, size=size)
        U3 = np.random.uniform(low=0.0, high=1.0, size=size)
        r     = _r * np.cbrt(U1)   # ← uses _r, not the frozen outer `rloc`
        theta = np.arccos(2*U2-1)
        phi   = 2*np.pi*U3
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.column_stack([x, y, z])

    valid_vectors = []
    _rloc_ = deepcopy(rloc)
    max_halvings = 20
    halvings = 0

    while len(valid_vectors) < no:
        aux_vector      = xyz_gen(no - len(valid_vectors), _rloc_)  
        distances       = np.linalg.norm(aux_vector, axis=1)
        inside_sphere   = aux_vector[distances <= _rloc_]
        _, nearest_indices = tree.query(inside_sphere)

        valid_mask   = Density[nearest_indices] > n_crit           
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)

        if len(valid_points) == 0:
            halvings += 1
            _rloc_ /= 2                         # ← expand outward, not inward
            warnings.warn(f"[snap={snap}] _rloc_ halved from {2*_rloc_} to {_rloc_}")

            if halvings >= max_halvings:
                warnings.warn(                  # ← fixed: message and category separated
                    f"[snap={snap}] At current snapshots, no cloud above {n_crit} cm-3 "
                    f"after {halvings} halvings (final _rloc_={_rloc_})",
                    RuntimeWarning
                )
                return None

    return np.array(deepcopy(valid_vectors))

@timing
def crs_path(*args, **kwargs):

    x_init = kwargs.get('x_init', None)
    n_crit = kwargs.get('n_crit', 1.0e+2)

    
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    m = x_init.shape[0]

    line      = np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((__alloc_slots__+1,m))
    densities = np.zeros((__alloc_slots__+1,m))
    pst_mask = np.zeros((m,)).astype(int) # one value for each

    line_rev=np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((__alloc_slots__+1,m))
    densities_rev = np.zeros((__alloc_slots__+1,m))
    pst_mask_rev = np.zeros((m,)).astype(int) # one value for each

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init 
    
    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    #densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3
    k=0

    mask2 = dens > n_crit
    un_masked2 = np.logical_not(mask2) # 1 if finished

    x_rev = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x_rev, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol_rev = Volume[cells]

    #densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens_rev = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    k=0
    k_rev=0

    mask2_rev = dens > n_crit
    un_masked2_rev = np.logical_not(mask2_rev)

    #while np.any(mask2) or np.any(mask2_rev): 
    while np.any(mask2) and (k + 1 < __alloc_slots__) or np.any(mask2_rev) and (k_rev + 1 < __alloc_slots__):

        mask2_rev = dens_rev > n_crit
        un_masked2_rev = np.logical_not(mask2_rev)

        if np.any(mask2_rev) and (k_rev + 1 < __alloc_slots__):
        
            x_rev_aux = x_rev[mask2_rev]

            x_rev_aux, bfield_aux_rev, dens_aux_rev, vol_rev = Euler_step(x_rev_aux, -1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
            dens_aux_rev = dens_aux_rev * gr_cm3_to_nuclei_cm3
            
            x_rev[mask2_rev] = x_rev_aux
            dens_rev[mask2_rev] = dens_aux_rev
            pst_mask_rev[mask2_rev] = bool(1)

            x_rev[un_masked2_rev] = 0
            dens_rev[un_masked2_rev] = 0
            pst_mask_rev[un_masked2_rev] = bool(0)
            
            #print(" alive lines? ",  np.any(mask2_rev), "k_rev + 1 < __alloc_slots__: ", k_rev + 1 < __alloc_slots__)

            line_rev[k_rev+1,mask2_rev,:] = x_rev_aux
            bfields_rev[k_rev+1,mask2_rev] = bfield_aux_rev
            densities_rev[k_rev+1,mask2_rev] = dens_aux_rev              

            k_rev += 1

        mask2 = dens > n_crit # above threshold
        un_masked2 = np.logical_not(mask2)
        
        if np.any(mask2) and (k + 1 < __alloc_slots__):

            x_aux = x[mask2]
            x_aux, bfield_aux, dens_aux, vol = Euler_step(x_aux, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
            dens_aux = dens_aux * gr_cm3_to_nuclei_cm3
            
            x[mask2]                   = x_aux
            dens[mask2]                = dens_aux
            pst_mask[mask2]            = bool(1)
            x[un_masked2]              = 0
            dens[un_masked2]           = 0
            pst_mask[un_masked2]       = bool(0)
            #print("k + 1 < __alloc_slots__: ", k + 1 < __alloc_slots__," alive lines? ",  np.any(mask2))
            line[k + 1, mask2, :]      = x_aux
            bfields[k + 1, mask2]      = bfield_aux
            densities[k + 1, mask2]    = dens_aux

            k += 1
        

    print(np.logical_not((np.any(mask2_rev) and (k_rev + 1 < __alloc_slots__))), np.logical_not((np.any(mask2) and (k + 1 < __alloc_slots__))))
    #threshold = threshold.astype(int)

    survivors_mask = np.logical_not(np.logical_and(pst_mask, pst_mask_rev))

    percentage_of_survivors = np.sum(survivors_mask)*100/survivors_mask.shape[0]

    print("Percentage of Survivors: ", percentage_of_survivors, " %")
    print("k = ", k, " k_rev = ", k_rev)
    
    nz_i    = k + 1
    nz_irev = k_rev + 1
    
    print(f"get_lines => threshold index for {__threshold__}cm-3: ", nz_i, nz_irev)
    print(f"get_lines => original shapes ({2*__alloc_slots__+1} to {nz_i + nz_irev - 1})")
    print(f"get_lines => p_r = {__alloc_slots__+1} to p_r = {nz_irev} for array with shapes ...")

    radius_vectors = np.append(line_rev[:nz_irev,:,:][::-1, :, :], line[1:nz_i,:,:], axis=0)
    magnetic_fields = np.append(bfields_rev[:nz_irev,:][::-1, :], bfields[1:nz_i,:], axis=0)
    numb_densities = np.append(densities_rev[:nz_irev,:][::-1, :], densities[1:nz_i,:], axis=0)

    #__alloc_slots__ = magnetic_fields.shape[0]

    print("Radius vector shape:", radius_vectors.shape)

    m = magnetic_fields.shape[1]

    #* 3.086e+18                                # from Parsec to cm
    #* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)
    
    path_column   = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) *pc_to_cm

    return radius_vectors, magnetic_fields, numb_densities, nz_irev, path_column, survivors_mask #p_r #, [threshold, threshold2, threshold_rev, threshold2_rev]

@timing
def line_of_sight(*args, **kwargs):

    x_init = kwargs.get('x_init', None)
    directions = kwargs.get('directions', fibonacci_sphere(20))
    n_crit = kwargs.get('n_crit', 1.0e+2)

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
    directions = np.tile(directions, (m0, 1))
    x_init = np.repeat(x_init, l0, axis=0)
    m = x_init.shape[0]
    l = directions.shape[0]
    """
    Now, a new feature that might speed the while loop, can be to double the size of all arrays
    and start calculating backwards and forwards simultaneously. This creates a more difficult condition
    for the 'mask', nevertheless, for a large array 'x_init' it may not be as different and it will definitely scale efficiently in parallel
    """

    line      = np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    line_rev=np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    densities = np.zeros((__alloc_slots__+1,m))
    densities_rev = np.zeros((__alloc_slots__+1,m))
    threshold = np.zeros((m,))
    threshold_rev = np.zeros((m,))

    line[0,:,:]     = x_init.copy()
    line_rev[0,:,:] = x_init.copy()
    x = x_init.copy()
    dummy, _0, densities[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Density_grad, Pos, VoronoiPos)
    vol = Volume[cells]
    #densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:].copy()
    k=0
    x_rev = x_init.copy()
    densities_rev[0,:], cells_rev = densities[0,:].copy(), cells.copy()
    vol_rev = Volume[cells_rev]
    #densities_rev[0,:] = densities_rev[0,:]
    dens_rev = densities_rev[0,:].copy()
    k_rev=0
    mask  = dens > n_crit# 1 if not finished
    un_masked = np.logical_not(mask) # 1 if finished
    mask_rev = dens_rev > n_crit
    un_masked_rev = np.logical_not(mask_rev)
   
    while np.any(mask) and (k + 1 < __alloc_slots__) or np.any(mask_rev) and (k_rev + 1 < __alloc_slots__):

        mask = dens > n_crit              # still alive?
        un_masked = np.logical_not(mask)  # any deaths?
        mask_rev = dens_rev > n_crit     
        un_masked_rev = np.logical_not(mask_rev)

        if np.any(mask) and (k + 1 < __alloc_slots__): # any_alive? and below_allocation?
            # continue if still lines alive and below allocation

            _, bfield, dens, vol = Heun_step(x, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)

            dens *= gr_cm3_to_nuclei_cm3
            
            vol[un_masked] = 0               # artifically make cell volume of finished lines equal to cero

            dx_vec = ((4 / 3) * vol / np.pi) ** (1 / 3)

            threshold += mask.astype(int)  # Increment threshold count only for values still above 100

            x += dx_vec[:, np.newaxis] * directions

            line[k+1,:,:]    = x
            densities[k+1,:] = dens

            k += 1

        if np.any(mask_rev) and (k_rev + 1 < __alloc_slots__):
            # continue if still lines alive and below allocation

            _, bfield_rev, dens_rev, vol_rev = Heun_step(x_rev, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
            dens_rev *= gr_cm3_to_nuclei_cm3
            
            vol_rev[un_masked_rev] = 0

            dx_vec = ((4 / 3) * vol_rev / np.pi) ** (1 / 3) 

            threshold_rev += mask_rev.astype(int)

            x_rev -= dx_vec[:, np.newaxis] * directions

            line_rev[k_rev+1,:,:]    = x_rev
            densities_rev[k_rev+1,:] = dens_rev

            k_rev += 1
        #print(k, k_rev)
        
    threshold = threshold.astype(int)
    threshold_rev = threshold_rev.astype(int)

    max_th     = np.max(threshold) + 1
    max_th_rev = np.max(threshold_rev) + 1 

    radius_vectors = np.append(line_rev[:max_th_rev,:,:][::-1, :, :], line[1:max_th,:,:], axis=0)*pc_to_cm
    numb_densities = np.append(densities_rev[:max_th_rev,:][::-1, :], densities[1:max_th,:], axis=0)
    column_densities = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) # list of 'm' column densities
    
    mean_columns = np.zeros(m0)
    median_columns= np.zeros(m0)
    i = 0
    
    #print("null means good", np.sum(np.where(numb_densities == 0)), "\n") # if this prints null then we alright

    for x in range(0, l0*m0, l0): 

        mean_columns[i] = np.mean(column_densities[x:x+l0])
        median_columns[i] = np.median(column_densities[x:x+l0])
        i += 1

    if False: # for a single point in space with l0 directions
        fig0, ax0 = plt.subplots()
        h = 0
        ax0.set_xlabel(r'''$s$ [cm]''', fontsize=16)
        ax0.set_ylabel(r'$n_g$ [cm$^{-3}$]', fontsize=16)
        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        
        for j in range(l0):

            s  = np.cumsum(np.linalg.norm(np.diff(radius_vectors[:, j, :], axis=0), axis=1))
            ng = numb_densities[1:, j] #* np.linalg.norm(np.diff(radius_vectors[:, j, :], axis=0), axis=1)

            ax0.plot(s, ng, '-',label=f'{j}')

        ax0.tick_params(axis='both')
        ax0.grid(True, which='both', alpha=0.3)
        ax0.legend(loc="upper left", fontsize=6)
        
        plt.title(r'Density [$n_g$] vs Distance [$s$]', fontsize=16)

        plt.savefig(f'./series/ng_vs_s.png',dpi=300)
        plt.close(fig0)

        fig1, ax1 = plt.subplots()

        # x0, ... 20 veces, x1, ..., 20 veces

        ax1.set_xlabel(r'''$s$ [cm]''', fontsize=16)
        ax1.set_ylabel(r'$N_{los}$ [cm$^{-2}$]', fontsize=16)
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        for j in range(l0):

            c  = np.cumsum(numb_densities[1:, j]*np.linalg.norm(np.diff(radius_vectors[:, j, :], axis=0), axis=1))
            ax1.plot(s, c, '-', label=f'{j}',linewidth=1)

        ax1.tick_params(axis='both')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend(loc="upper left", fontsize=6)
        
        plt.title(r'Column [$N_{los}$] vs Distance [$s$]', fontsize=16)

        plt.savefig(f'./series/c_vs_s.png',dpi=300)
        plt.close(fig1)

    return radius_vectors, numb_densities, mean_columns, median_columns

def match_files_to_data(__input_case__):
    print("__input_case__: ", __input_case__)
    
    if 'ideal' in __input_case__:
        subdirectory = 'ideal_mhd'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))
        file_xyz       = f'./util/{__input_case__}_cloud_trajectory.txt'
    elif 'amb' in __input_case__:
        subdirectory = 'ambipolar_diffusion'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))
        file_xyz       = f'./util/{__input_case__}_cloud_trajectory.txt'

    assert os.path.exists(file_xyz), f"[{file_xyz} cloud data not present]"

    print(file_xyz)
    print(file_hdf5)
 
    with open(file_xyz, mode='r') as file:
        clst = []; dlst = []; tlst = []; slst = []
        csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
        next(csv_reader)  # Skip the header row
        
        for row in csv_reader:
            #print(row[0],row[1], np.log10(float(row[8])))
            clst.append([np.float64(row[2]),np.float64(row[3]),np.float64(row[4])])
            dlst.append(np.float64(row[8]))
            tlst.append(np.float64(row[1]))
            slst.append(int(row[0]))
            if (int(row[0]) < int(__start_snap__)):
                break

    assert len(slst) != 0, "Error accessing cloud data"

    clst, dlst, tlst, slst = [np.array(arr, dtype=FloatType) for arr in (clst[::-1], dlst[::-1], tlst[::-1], slst[::-1])]

    validated = np.logical_not(dlst); white_list = []

    for i, f in enumerate(file_hdf5):
        snap = int(f.split('/')[-1].split('.')[-2][-3:])
        print(f"{snap} data available? ", snap in slst)
        if snap in slst:
            #print(int(f.split('.')[0][-3:]), np.where(slst == int(f.split('.')[0][-3:])))    
            validated[np.where(slst == snap)] = True
            white_list += [True]
        else:
            white_list += [False]

    clst, dlst, tlst, slst = [arr[validated] for arr in (clst, dlst, tlst, slst)]

    file_hdf5 = np.sort(file_hdf5[np.array(white_list)][::-1])

    return clst, dlst, tlst, slst, file_hdf5

def describe(data, band=False, percent=False):

    # data shape: (50, 1000)
    mean_   = np.mean(data, axis=0)
    median_ = np.median(data, axis=0)
    skew_   = skew(data, axis=0)
    kurt_   = kurtosis(data, axis=0)

    if band:
        sigma1_low, sigma1_high = np.percentile(data, [16, 84], axis=0) # 1-sigma band
        return sigma1_low, sigma1_high
    if percent:
        p_25    = np.percentile(data, 25, axis=0)
        p_10    = np.percentile(data, 10, axis=0)
        p_5     = np.percentile(data, 5, axis=0)
        return p_25, p_10, p_5
    return mean_, median_, skew_, kurt_

def declare_globals_and_constants():

    # global variables that can be modified from anywhere in the code
    global __rloc__, __sample_size__, __input_case__, __start_snap__
    global __dense_cloud__,__threshold__, __alloc_slots__
    global FloatType, FloatType2, IntType

    # amb:   t > 3.0 Myrs snap > 225    
    # ideal: t > 3.0 Myrs snap > 270
    __start_time__  = 3.0 # Myrs
    if __input_case__ =='ideal':
        __start_snap__  = '270'

    if __input_case__ =='amb':
        __start_snap__  = '225'

    # rename
    FloatType          = np.float64
    FloatType2         = np.float128
    IntType            = np.int32

    os.makedirs("series", exist_ok=True)

    return None

input_file = sys.argv[1]
print("IC file is: ", input_file)
print("within inputs/ dir: ", input_file.split('/')[-1])

FLAG0 = "-l" 
FLAG1 = "-e" 
config = {}
with open(input_file, 'r') as f:
    exec(f.read(), {}, config)

globals().update(config) # This injects every key as a variable in your script

input_file = input_file.split('/')[-1]
print("ID of input file [type][number]: ", str(input_file.split('.')[0][0] + input_file.split('.')[0][-1]), flush=True)

print(f"\n[{sys.argv[0]}]: started running...\n", flush=True)


if __name__=='__main__':
    # declares global variables
    declare_globals_and_constants()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        clst, dlst, tlst, slst, file_hdf5 = match_files_to_data(__input_case__)
        _id_ = str(input_file.split('.')[0][0] + input_file.split('.')[0][-1])
        if FLAG1 in sys.argv:
            _id_ = str(input_file.split('.')[0][0] + input_file.split('.')[0][-3:])
        print("ID of series.py run is", _id_)
    else:
        clst = dlst = tlst = slst = file_hdf5 = None
        _id_ = None

    clst, dlst, tlst, slst, file_hdf5, _id_ = comm.bcast(
        (clst, dlst, tlst, slst, file_hdf5, _id_), root=0
    )
    
    assert len(file_hdf5) == len(clst) == len(tlst), "Arrays must all have the same length"

    survivors_fraction = np.zeros(file_hdf5.shape[0])
    
    df_stats = dict()
    df_fields= dict()

    print(_id_)
    for each in range(rank, len(file_hdf5), size):
        filename = file_hdf5[each]
        center   = clst[each, :]
        _time    = tlst[each]
        print(filename, flush = True)

        snap = int(filename.split('.')[0][-3:])
        data = h5py.File(filename, 'r')
        __Boxsize__ = data['Header'].attrs['BoxSize']

        Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
        Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)*gr_cm3_to_nuclei_cm3
        VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
        Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
        Density_grad = np.zeros((len(Density), 3))
        Volume   = np.asarray(data['PartType0']['Masses'], dtype=FloatType)/Density

        VoronoiPos-=center
        Pos       -=center    

        for dim in range(3):
            pos_from_center = Pos[:, dim]
            too_high = pos_from_center > __Boxsize__ / 2
            too_low  = pos_from_center < -__Boxsize__ / 2
            Pos[too_high, dim] -= __Boxsize__
            Pos[too_low,  dim] += __Boxsize__

        table_data = [
            ["__curr_snap__", filename.split('/')[-1]],
            ["__cores_avail__", os.cpu_count()],
            ["__rloc__", f"{__rloc__}" + " pc"],
            ["__threshold__", f"{__threshold__}"+ " cm^-3"],
            ["__dense_cloud__", f"{__dense_cloud__}"+ " cm^-3"],
            ["__alloc_slots__", __alloc_slots__],
            ["__sample_size__", __sample_size__]
            ]

        print(tabulate(table_data, headers=["Property", "Value"], tablefmt="grid"), '\n', flush=True)

        stats_dict = {
            "time": _time, 
            "x_input": np.array([]),
            "n_rs": np.array([]),
            "B_rs": np.array([]),
            "n_path": np.array([]),
            "n_los0": np.array([]),
            "n_los1": np.array([]),
            "surv_fraction": np.array([]),
            "r_u": np.array([]),
            "r_l": np.array([])
        }

        tree = cKDTree(Pos)

        try:
            x_input    = uniform_in_3d_tree_dependent(tree, __sample_size__, rloc=__rloc__, n_crit=__dense_cloud__)   
        except Exception as e:
            print(f"[Snap] snap {snap}: skipping — {type(e).__name__}: {e}", flush=True)
            data.close()  
            df_stats[str(snap)]  = stats_dict
            continue


        if x_input is None:
            print(f"[Snap] snap {snap}: skipping", flush=True)
            data.close()
            df_stats[str(snap)]  = stats_dict
            continue

        directions=fibonacci_sphere(20)        # dodecahedron (12 faces), and icosahedron (20 faces)
        try:
            __0, __1, mean_column, median_column = line_of_sight(x_init=x_input, directions=directions, n_crit=__threshold__)
            radius_vectors, magnetic_fields, numb_densities, follow_index, path_column, survivors1 = crs_path(x_init=x_input, n_crit=__threshold__)
            assert np.any(numb_densities > __threshold__), f"No values above threshold {__threshold__} cm-3"
        except:
            print(f"[LOS/CRS] Invalid result from intergration: {snap}: skipping", flush=True)
            data.close()
            df_stats[str(snap)]  = stats_dict
            continue

        print("__alloc_slots__: ", __alloc_slots__, flush=True)
        print("__used_slots__ : ",__0.shape, flush=True)

        if np.log10(__threshold__) < 2: 
            r_u, n_rs, B_rs, survivors2 = eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__*10)
            r_l, _1, _2, _3 = eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__)
        else:
            r_u, n_rs, B_rs, survivors2 = eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__)
            r_l, _1, _2, _3 = eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__) # make redundant
        
        survivors = np.logical_and(survivors1, survivors2)

        print(np.sum(survivors)/survivors.shape[0], " Survivor fraction", flush=True)

        survivors_fraction[each] = np.sum(survivors)/survivors.shape[0]
        u_input         = x_input[np.logical_not(survivors),:] # pc
        x_input         = x_input[survivors,:]                 # pc
        radius_vectors  = radius_vectors[:, survivors, :]      # pc
        numb_densities  = numb_densities[:, survivors]         # cm-3
        magnetic_fields = magnetic_fields[:, survivors]*gauss_code_to_gauss_cgs # Gauss CGS
        mean_column     = mean_column[survivors]               # cm-2
        median_column   = median_column[survivors]             # cm-2
        path_column     = path_column[survivors]               # cm-2
        n_rs            = n_rs[survivors]                      # cm-3
        B_rs            = B_rs[survivors]*gauss_code_to_gauss_cgs # Gauss CGS
        r_u             = r_u[survivors]                       # Adim
        r_l             = r_l[survivors]                       # Adim
        

        mean_r_u, median_r_u, skew_r_u, kurt_r_u = describe(r_u)
        mean_r_l, median_r_l, skew_r_l, kurt_r_l = describe(r_l)

        stats_dict = {
            "time": _time, 
            "x_input": x_input,
            "n_rs": n_rs,
            "B_rs": B_rs,
            "n_path": path_column,
            "n_los0": mean_column,
            "n_los1": median_column,
            "surv_fraction": survivors_fraction[each],
            "r_u": r_u,
            "r_l": r_l
        }

        df_stats[str(snap)]  = stats_dict

        if FLAG0 in sys.argv:
            field_dict = {
                "time": _time,
                "directions": directions,
                "x_input": x_input,
                "B_s": magnetic_fields,
                "r_s": radius_vectors,
                "n_s": numb_densities
            }

            df_fields[str(snap)]  = field_dict


        if 'Pos' in globals():
            print("\nPos is global", flush=True)

        # at the end of the loop, drop all that will be reasigned, to avoid memory overflow
        del tree, __0, __1, _1, _2, _3
        del radius_vectors, magnetic_fields, numb_densities
        del Pos, VoronoiPos, Bfield, Density, Volume, Density_grad
        gc.collect()
        data.close()

        with open(f'./series/tmp_{_id_}_rank{rank}.pkl', 'wb') as f:
            pickle.dump(df_stats, f)
            f.flush()
            os.fsync(f.fileno())
    
    comm.Barrier()
    if rank == 0:
        import asyncio
        import glob
    
        expected = comm.Get_size()

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(RuntimeError))    
        async def merge_and_save(_id_, __dense_cloud__):
            loop = asyncio.get_event_loop()

            stat_files = sorted(glob.glob(f'./series/tmp_{_id_}_rank*.pkl'))
            print(stat_files)

            # Read all rank files concurrently
            def load_pickle(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)

            stat_tasks = [loop.run_in_executor(None, load_pickle, f) for f in stat_files]

            all_stats = await asyncio.gather(*stat_tasks)

            # Merge stats
            merged_stats = {}
            for d in all_stats:
                merged_stats.update(d)

            # Save final output
            df = pd.DataFrame.from_dict(merged_stats, orient='index')\
                .reset_index().rename(columns={'index': 'snapshot'})
            df.to_pickle(f'./series/data_{int(np.log10(__dense_cloud__))}{_id_}.pkl')

            print(f"Merged {len(stat_files)} rank files successfully.", flush=True)

            for f in stat_files:
                #os.remove(f)
                print(f)

        asyncio.run(merge_and_save(_id_, __dense_cloud__))
        comm.Barrier()

    elapsed_time =time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"Elapsed time: {hours}h {minutes}m {seconds}s")
    
