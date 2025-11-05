import csv, glob, os, sys, time, h5py
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from copy import deepcopy

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.spatial import cKDTree

from src.library import *
from stats import *

@timing_decorator
def uniform_in_3d_tree_dependent(tree, no, rloc=1.0, n_crit=threshold):

    valid_vectors = []
    _rloc_ = deepcopy(rloc)
    while len(valid_vectors) < no:
        aux_vector, _ = xyz_gen(no - len(valid_vectors)) # [[x,y,z], [x,y,z], ...] <= np array
        distances = np.linalg.norm(aux_vector, axis=1)
        inside_sphere = aux_vector[distances <= _rloc_]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > n_crit
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
        print(len(valid_vectors))
        if len(valid_vectors) == 0:
            _rloc_ /=2
            Warning(f"[snap={snap}] _rloc_ halved from {_rloc_*2} to {_rloc_}")
        
        if _rloc_ < 1.0e-5:
            raise LookupError(f"[snap={snap}] At current snapshots, no cloud above {n_crit} cm-3")
    
    
    if False:
        Xs = np.array(deepcopy(valid_vectors))
        fig, ax = plt.subplots()
        ax.scatter(Xs[:,0], Xs[:,1])
        plt.savefig('./series/Xs.png')
        plt.close(fig)
        return Xs

    return np.array(deepcopy(valid_vectors))

@timing_decorator
def crs_path(*args, **kwargs):
    x_init = kwargs.get('x_init', None)
    n_crit = kwargs.get('n_crit', 1.0e+2)

    
    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))
    pst_mask = np.zeros((m,)).astype(int) # one value for each

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
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
    while np.any(mask2) and (k + 1 < N) or np.any(mask2_rev) and (k_rev + 1 < N):

        mask2_rev = dens_rev > n_crit
        un_masked2_rev = np.logical_not(mask2_rev)

        if np.any(mask2_rev) and (k_rev + 1 < N):
        
            x_rev_aux = x_rev[mask2_rev]

            x_rev_aux, bfield_aux_rev, dens_aux_rev, vol_rev = Euler_step(x_rev_aux, -1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
            dens_aux_rev = dens_aux_rev * gr_cm3_to_nuclei_cm3
            
            x_rev[mask2_rev] = x_rev_aux
            dens_rev[mask2_rev] = dens_aux_rev
            pst_mask_rev[mask2_rev] = bool(1)

            x_rev[un_masked2_rev] = 0
            dens_rev[un_masked2_rev] = 0
            pst_mask_rev[un_masked2_rev] = bool(0)
            
            #print(" alive lines? ",  np.any(mask2_rev), "k_rev + 1 < N: ", k_rev + 1 < N)

            line_rev[k_rev+1,mask2_rev,:] = x_rev_aux
            bfields_rev[k_rev+1,mask2_rev] = bfield_aux_rev
            densities_rev[k_rev+1,mask2_rev] = dens_aux_rev              

            k_rev += 1

        mask2 = dens > n_crit # above threshold
        un_masked2 = np.logical_not(mask2)
        
        if np.any(mask2) and (k + 1 < N):

            x_aux = x[mask2]
            x_aux, bfield_aux, dens_aux, vol = Euler_step(x_aux, 1.0, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
            dens_aux = dens_aux * gr_cm3_to_nuclei_cm3
            
            x[mask2]                   = x_aux
            dens[mask2]                = dens_aux
            pst_mask[mask2]            = bool(1)
            x[un_masked2]              = 0
            dens[un_masked2]           = 0
            pst_mask[un_masked2]       = bool(0)
            #print("k + 1 < N: ", k + 1 < N," alive lines? ",  np.any(mask2))
            line[k + 1, mask2, :]      = x_aux
            bfields[k + 1, mask2]      = bfield_aux
            densities[k + 1, mask2]    = dens_aux

            k += 1
        
        #print(k, k_rev)

    print(np.logical_not((np.any(mask2_rev) and (k_rev + 1 < N))), np.logical_not((np.any(mask2) and (k + 1 < N))))
    #threshold = threshold.astype(int)

    survivors_mask = np.logical_not(np.logical_and(pst_mask, pst_mask_rev))

    percentage_of_survivors = np.sum(survivors_mask)*100/survivors_mask.shape[0]

    print("Percentage of Survivors: ", percentage_of_survivors, " %")
    
    nz_i    = k + 1
    nz_irev = k_rev + 1
    
    print(f"get_lines => threshold index for {threshold}cm-3: ", nz_i, nz_irev)
    print(f"get_lines => original shapes ({2*N+1} to {nz_i + nz_irev - 1})")
    print(f"get_lines => p_r = {N+1} to p_r = {nz_irev} for array with shapes ...")

    radius_vectors = np.append(line_rev[:nz_irev,:,:][::-1, :, :], line[1:nz_i,:,:], axis=0)
    magnetic_fields = np.append(bfields_rev[:nz_irev,:][::-1, :], bfields[1:nz_i,:], axis=0)
    numb_densities = np.append(densities_rev[:nz_irev,:][::-1, :], densities[1:nz_i,:], axis=0)

    #N = magnetic_fields.shape[0]

    print("Radius vector shape:", radius_vectors.shape)

    m = magnetic_fields.shape[1]

    #* 3.086e+18                                # from Parsec to cm
    #* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)
    
    path_column   = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) *pc_to_cm

    return radius_vectors, magnetic_fields, numb_densities, nz_irev, path_column, survivors_mask #p_r #, [threshold, threshold2, threshold_rev, threshold2_rev]

@timing_decorator
def line_of_sight(*args, **kwargs):
    # Unpack positional arguments
    #Bfield, Density, Mass, Bfield_grad, Density_grad = args[:5]

    # Optional positional/keyword arguments
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

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    densities = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
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
   
    while np.any(mask) and (k + 1 < N) or np.any(mask_rev) and (k_rev + 1 < N):

        mask = dens > n_crit              # still alive?
        un_masked = np.logical_not(mask)  # any deaths?
        mask_rev = dens_rev > n_crit     
        un_masked_rev = np.logical_not(mask_rev)

        if np.any(mask) and (k + 1 < N): # any_alive? and below_allocation?
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

        if np.any(mask_rev) and (k_rev + 1 < N):
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
    
    print("null means good", np.sum(np.where(numb_densities == 0)), "\n") # if this prints null then we alright

    for x in range(0, l0*m0, l0): 

        mean_columns[i] = np.mean(column_densities[x:x+l0])
        median_columns[i] = np.median(column_densities[x:x+l0])
        i += 1

    if True: # for a single point in space with l0 directions
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

        plt.savefig(f'./stats/ng_vs_s.png',dpi=300)
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

        plt.savefig(f'./stats/c_vs_s.png',dpi=300)
        plt.close(fig1)

    return radius_vectors, numb_densities, mean_columns, median_columns

@timing_decorator
def match_files_to_data(__input_case__):
    
    if __input_case__ in 'ideal_mhd':
        subdirectory = 'ideal_mhd'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))
        file_xyz       = f'./{__input_case__}_cloud_trajectory.txt'
    elif __input_case__ in 'ambipolar_diffusion':
        subdirectory = 'ambipolar_diffusion'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))
        file_xyz       = f'./{__input_case__}_cloud_trajectory.txt'

    assert os.path.exists(file_xyz), f"[{file_xyz} cloud data not present]"
 
    with open(file_xyz, mode='r') as file:
        clst = []; dlst = []; tlst = []; slst = []
        csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
        next(csv_reader)  # Skip the header row
        
        for row in csv_reader:
            #print(row[0],row[1], np.log10(float(row[8])))
            if (int(row[0]) < int(__start_snap__)):
                break
            clst.append([np.float64(row[2]),np.float64(row[3]),np.float64(row[4])])
            dlst.append(np.float64(row[8]))
            tlst.append(np.float64(row[1]))
            slst.append(int(row[0]))
        
    assert len(slst) != 0, "Error accessing cloud data"

    clst, dlst, tlst, slst = [np.array(arr, dtype=FloatType) for arr in (clst[::-1], dlst[::-1], tlst[::-1], slst[::-1])]

    validated = np.logical_not(dlst); white_list = []

    for i, f in enumerate(file_hdf5):
        if int(f.split('.')[0][-3:]) in slst: 
            #print(int(f.split('.')[0][-3:]), np.where(slst == int(f.split('.')[0][-3:])))    
            validated[np.where(slst == int(f.split('.')[0][-3:]))] = True
            white_list += [True]
        else:
            white_list += [False]

    clst, dlst, tlst, slst = [arr[validated] for arr in (clst, dlst, tlst, slst)]

    file_hdf5 = np.sort(file_hdf5[np.array(white_list)][::-1])

    return clst, dlst, tlst, slst, file_hdf5

@timing_decorator
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

def use_lock_and_save(path):
    from filelock import FileLock
    """
    Time ──────────────────────────────▶

    Script 1:  ──[Acquire Lock]─────[Create/Append HDF5]─────[Release Lock]───

    Script 2:             ──[Wait for Lock]─────────────[Append HDF5]─────[Release Lock]───
    """
    lock = FileLock(path + ".lock")

    with lock:  # acquire lock (waits if another process is holding it)

        pass
        # Use this to create and update a dataframe

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

def larson_width_line_relation(**kwargs):
    velocities = kwargs.get('Velocities', None)
    data = kwargs.get('data', None)
    
    if data:
        #data = np.array([
        #    (0.1590312239231974, 0.49723685684431057),
        #    (0.04564133526179595, 0.3094255487061252),
        #    (0.08250841915338421, 0.3874978892146356)
        #])

        L, sigma_v = data[:, 0], data[:, 1]

        logL = np.log10(L)
        logsig = np.log10(sigma_v)
        coeffs = np.polyfit(logL, logsig, 1)
        alpha = coeffs[0]
        A = 10 ** coeffs[1]
        
        print(f"Power-law fit: sigma_v = {A:.4f} * L^{alpha:.4f}")

        # --- Plot ---
        plt.figure(figsize=(5,4))
        plt.loglog(L, sigma_v, 'o', color='royalblue', label='Data')
        plt.loglog(L, A * L**alpha, '--', color='darkorange',
                   label=fr'Fit: $\sigma_v = {A:.2f}L^{{{alpha:.2f}}}$')
        plt.xlabel('L (pc)')
        plt.ylabel(r'$\sigma_v$ (km/s)')
        plt.title('Larson Width–Size Relation')
        plt.grid(True, which='both', ls='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./larson_width_size.png')
        return 0

    """
    Velocities must be provided in km/s
    L will be returned in 
    """
    c = (Pos[:, 0]*Pos[:, 0] + Pos[:, 1]*Pos[:, 1] + Pos[:, 2]*Pos[:, 2] < __rloc__*__rloc__)
    d = Density * gr_cm3_to_nuclei_cm3 > __dense_cloud__
    cd = np.logical_and(c,d)
    vel_mags = np.linalg.norm(velocities[cd], axis=1)  # magnitude of velocities within cloud
    sigma_vel = np.std(vel_mags)
    p = 0.38
    sigma_scale = 1.0 # km/s
    L_scale = 1.0 # pc
    L = L_scale*(sigma_vel/sigma_scale)**(1/p)

    # remember that L represents a diameters or width of the cloud
    return L, sigma_vel # width of cloud, velocity dispersion

@timing_decorator
def evolution_descriptors():
    df_factor = pd.read_pickle('./series/r_stats.pkl')
    df_column = pd.read_pickle('./series/c_stats.pkl')

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

    df_factor.plot(x='time', y='mean_r_u', ax=ax0, label=f'mean')
    df_factor.plot(x='time', y='median_r_u', ax=ax0, label=f'median')
    ax0.set_xlabel('$t - t_{G-ON}$ [Myr]', fontsize=16)
    ax0.set_ylabel('$R$ Factor', fontsize=16)
    ax0.legend(fontsize=16)
    ax0.grid(True)

    df_factor.plot(x='time', y='skew_r_u', ax=ax1, label=f'skew')
    df_factor.plot(x='time', y='kurt_r_u', ax=ax1, label=f'kurt')
    ax1.set_xlabel('$t - t_{G-ON}$ [Myr]', fontsize=16)
    ax1.legend(fontsize=16)
    ax1.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for legend
    plt.savefig(f'./series/descriptors_{__input_case__}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        df_column['n_path'].tolist(),       # convert Series of arrays -> list of arrays
        positions=df_column['snapshot'],    # x-axis positions
        widths=2,                           # adjust width
        patch_artist=True,                  # color the boxes
        medianprops=dict(color='red')       # median line color
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("ratio0")
    ax.set_yscale("log")
    ax.set_title("Evolution of ratio0 over Time")

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig(f'./series/ratio0_{__input_case__}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

@timing_decorator
def declare_globals_and_constants():

    # global variables that can be modified from anywhere in the code
    global __rloc__, __sample_size__, __input_case__, __start_snap__, __alloc_slots__, __dense_cloud__,__threshold__, N
    global Pos, VoronoiPos, Bfield, Mass, Density, Bfield_grad, Density_grad, Volume
    global flag, FloatType, FloatType2, IntType
    
    # immutable objects, use type hinting to debug if error
    N               = 2_500
    __rloc__        = 0.1
    __sample_size__ = 1_000
    __input_case__  = 'ideal'
    __start_snap__  = sys.argv[1]
    __alloc_slots__ = 2_000
    __dense_cloud__ = 1.0e+2
    __threshold__   = 1.0e+2

    # rename
    FloatType                = np.float64
    FloatType2                = np.float128
    IntType                  = np.int32

    # for timing
    start_time = time.time()

    # output directory
    os.makedirs("series", exist_ok=True)

    # flag to import data
    flag = True
    return None


if __name__=='__main__':
    declare_globals_and_constants()   

    # coordinates, cell center density, time and snapshot of evolving cloud
    clst, dlst, tlst, slst, file_hdf5 = match_files_to_data(__input_case__)
    survivors_fraction = np.zeros(file_hdf5.shape[0])
    
    df_r_stats = dict()
    df_c_stats = dict()

    for each, filename in enumerate(file_hdf5):

        snap = int(filename.split('.')[0][-3:])
        data = h5py.File(filename, 'r')
        __Boxsize__ = data['Header'].attrs['BoxSize']

        Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
        Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)*gr_cm3_to_nuclei_cm3
        VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
        #Velocities = np.asarray(data['PartType0']['Velocities'], dtype=FloatType)
        Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
        Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
        Bfield_grad = np.zeros((len(Pos), 9))
        Density_grad = np.zeros((len(Density), 3))
        Volume   = Mass/Density
        center = clst[each,:]
        VoronoiPos-=center
        Pos       -=center    
        _time = tlst[each]

        for dim in range(3):
            pos_from_center = Pos[:, dim]
            too_high = pos_from_center > __Boxsize__ / 2
            too_low  = pos_from_center < -__Boxsize__ / 2
            Pos[too_high, dim] -= __Boxsize__
            Pos[too_low,  dim] += __Boxsize__

        from tabulate import tabulate
        table_data = [
            ["__snaps_interval__", filename.split('/')[-1]],
            ["__cores_avail__", os.cpu_count()],
            ["__alloc_slots__", 2*N],
            ["__rloc__", __rloc__],
            ["__sample_size__", __sample_size__],
            ["__Boxsize__", __Boxsize__]
            ]
        
        print(tabulate(table_data, headers=["Property", "Value"], tablefmt="grid"), '\n')
        
        tree = cKDTree(Pos)

        try:
            #x_input = np.vstack([uniform_in_3d(__sample_size__, __rloc__, n_crit=__dense_cloud__), np.array([0.0,0.0,0.0])])
            #print(__sample_size__, __rloc__, __dense_cloud__) # uniform_in_3d_tree_dependent
            x_input    = uniform_in_3d_tree_dependent(tree, __sample_size__, rloc=__rloc__, n_crit=__dense_cloud__)    
        except:
            raise ValueError("[Error] Faulty generation of 'x_input'")

        directions=fibonacci_sphere(10)
        __0, __1, mean_column, median_column = line_of_sight(x_init=x_input, directions=directions, n_crit=__threshold__)
        radius_vectors, magnetic_fields, numb_densities, follow_index, path_column, survivors1 = crs_path(x_init=x_input, n_crit=__threshold__)

        assert np.any(numb_densities > __threshold__), f"No values above threshold {__threshold__} cm-3"

        data.close()

        r_u, n_rs, B_rs, survivors2 = eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__)
        r_l, _1, _2, _3 = eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__) # not needed

        survivors = np.logical_and(survivors1, survivors2)

        print(np.sum(survivors)/survivors.shape[0], " Survivor fraction")  

        survivors_fraction[each] = np.sum(survivors)/survivors.shape[0]
        u_input         = x_input[np.logical_not(survivors),:] 
        x_input         = x_input[survivors,:]
        radius_vectors  = radius_vectors[:, survivors, :]
        numb_densities  = numb_densities[:, survivors]
        magnetic_fields = magnetic_fields[:, survivors]
        mean_column     = mean_column[survivors]
        median_column   = median_column[survivors]
        path_column     = path_column[survivors]
        n_rs            = n_rs[survivors]
        B_rs            = B_rs[survivors] 
        r_u             = r_u[survivors]
        r_l             = r_l[survivors]

        #distance = np.linalg.norm(x_input, axis=1)*pc_to_cm

        mean_r_u, median_r_u, skew_r_u, kurt_r_u = describe(r_u)
        mean_r_l, median_r_l, skew_r_l, kurt_r_l = describe(r_l)

        """
        smth
        """
        c_stats_dict = {
            "time": _time,
            "x_input": x_input,
            "n_rs": n_rs,
            "B_rs": B_rs,
            "n_path": path_column,
            "ratio0": mean_column/path_column,
            "ratio1": median_column/path_column
        }
        df_c_stats[str(snap)]  = c_stats_dict

        r_stats_dict = {
            "time": _time,
            "surv_fraction": survivors_fraction[each],
            "mean_r_u": mean_r_u,
            "median_r_u": median_r_u,
            "skew_r_u": skew_r_u,
            "kurt_r_u": kurt_r_u,
            "mean_r_l": mean_r_l,
            "median_r_l": median_r_l,
            "skew_r_l": skew_r_l,
            "kurt_r_l": kurt_r_l
        }

        df_r_stats[str(snap)]  = r_stats_dict # multiindex dataframe
        

        if 'Pos' in globals():
            print("\nPos is global")
            
        del tree

        if each == 0:
            continue
        else:
            import src.library
            src.library._cached_tree = None
            src.library._cached_pos = None

    # For more information regarding the interpretation of skewness and kurtosis 
    # https://www.statology.org/how-to-report-skewness-kurtosis/

    # Loop through each plot and generate/save independently

    # to save dataframes into values other than strings it must be pickleized
    df_r = pd.DataFrame.from_dict(df_r_stats, orient='index').reset_index().rename(columns={'index': 'snapshot'})
    df_c = pd.DataFrame.from_dict(df_c_stats, orient='index').reset_index().rename(columns={'index': 'snapshot'})

    df_r.to_pickle('./series/r_stats.pkl')
    df_c.to_pickle('./series/c_stats.pkl')

    print("Saved r_stats.pkl and c_stats.pkl with full NumPy arrays intact.")

    print(df_r.describe())
    print(df_c.describe())

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    df_r.plot(x='time', y='mean_r_u', ax=ax0, label=f'mean')
    df_r.plot(x='time', y='median_r_u', ax=ax0, label=f'median')
    ax0.set_xlabel('$t - t_{G-ON}$ [Myr]', fontsize=16)
    ax0.set_ylabel('$R$ Factor', fontsize=16)
    #ax0.set_title('Ideal MHD', fontsize=16)
    ax0.legend(fontsize=16)
    ax0.grid(True)

    df_r.plot(x='time', y='skew_r_u', ax=ax1, label=f'skew')
    df_r.plot(x='time', y='kurt_r_u', ax=ax1, label=f'kurt')
    ax1.set_xlabel('$t - t_{G-ON}$ [Myr]', fontsize=16)
    ax1.set_ylabel('$R$ Factor', fontsize=16)
    #ax1.set_title('Non-ideal MHD - AD', fontsize=16)
    ax1.legend(fontsize=16)
    ax1.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for legend
    plt.savefig(f'series/descriptors_{__input_case__}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot and save n^mean_los / n_path and n^median_los / n_path
    # plot and save n^mean_los / n_path and n^median_los / n_path

    # plot together
    # plot and save Reduction factor over n_g using log10(n_g(p)/n_{ref}) < 1/8
    # mean, median, 1sigma band and 25, 10, 5 th percentiles
    # n_ref must be between 10^2 and max value of density in the 495 snapshot (10^4)
