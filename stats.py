import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import functools
import numpy as np
from src.library import *
import csv, glob, os, sys, time
import h5py

start_time = time.time()

FloatType = np.float64
IntType = np.int32

N               = 2_500

if len(sys.argv)>4:
    rloc          = float(sys.argv[1])
    max_cycles    = int(sys.argv[2]) 
    case          = str(sys.argv[3]) 
    num_file      = str(sys.argv[4]) 
else: #  python3 stats.py 0.1 2000 ideal 480
    rloc            = 0.1
    max_cycles      = 500
    case            = 'ideal'
    num_file        = '430'

dense_cloud = 1.0e+2
threshold = 1.0e+2

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

                
    rdcut = []
    for i in range(0,Npoints):
        if density_data[i] > n_min:
            rdcut = rdcut + [reduction_data[i]]

    if True:
        fig = plt.figure(figsize = (12, 6))
        ax1 = fig.add_subplot(121)
        ax1.hist(rdcut, round(np.sqrt(Npoints)))
        ax1.set_xlabel('Reduction factor', fontsize = 16)
        ax1.set_ylabel('number', fontsize = 16)
        ax1.set_title(f't = {time_value}', fontsize = 16)
        plt.setp(ax1.get_xticklabels(), fontsize = 16)
        plt.setp(ax1.get_yticklabels(), fontsize = 16)
        ax2 = fig.add_subplot(122)
        l1, = ax2.plot(x_n, mean_vec)
        l2, = ax2.plot(x_n, median_vec)
        l3, = ax2.plot(x_n, ten_vec)
        try:
            ax2.scatter(density_data, reduction_data, alpha = 0.5, color = 'grey')
        except:
            pass
        plt.legend((l1, l2, l3), ('mean', 'median', '10$^{\\rm th}$ percentile'), loc = "lower right", prop = {'size':14.0}, ncol =1, numpoints = 5, handlelength = 3.5)
        plt.xscale('log')
        plt.ylim(0.25, 1.05)
        ax2.set_ylabel('Reduction factor', fontsize = 16)
        ax2.set_xlabel('gas density (hydrogens per cm$^3$)', fontsize = 16)
        ax2.set_title(f'f(R=1) = {fraction}', fontsize = 16)
        plt.setp(ax2.get_xticklabels(), fontsize = 16)
        plt.setp(ax2.get_yticklabels(), fontsize = 16)
        fig.subplots_adjust(left = .1)
        fig.subplots_adjust(bottom = .15)
        fig.subplots_adjust(top = .98)
        fig.subplots_adjust(right = .98)
        fig.tight_layout()

        #plt.savefig('histograms/pocket_statistics_ks.pdf')
        plt.savefig(f'./images/reduction/pocket_stats{bound}.png', dpi=300)
    return None

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

    if False:
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
            axs.set_yscale('log')
            axs.set_xlabel("Index")
            axs.set_ylabel(label)
            axs.set_title(f"{label} Shape")
            axs.legend()
            axs.grid(True)

        # Plot both subplots
        plot_field(axs_numb, numb, "Density")
        plot_field(axs_bfield, bfield, "Magnetic Field")

        plt.tight_layout()
        plt.savefig('./images/columns/mosaic.png')
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def eval_reduction(field, numb, follow_index, threshold):

    R10      = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    #print("_, m = field.shape => ", _, m)

    flag = False
    filter_mask = np.ones(m).astype(bool)
    dead = 0
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
        from collections import Counter
        most_common_value, count = Counter(bfield10.ravel()) .most_common(1)[0]
    
        if count > 20:
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)   
            flag = True
            print(f"Most common value: {most_common_value} (appears {count} times): R = ", R)
            filter_mask[i] = False
            if False:
                plt.plot(bfield10)
                plt.savefig(f'./see{i}.png')
                plt.close()
            dead +=1
            continue     

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
    print(dead)

    return np.array(R10), np.array(Numb100), np.array(B100), filter_mask

@timing_decorator
def crs_path(x_init=np.array([0,0,0]),ncrit=threshold):
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

    mask2 = dens > ncrit
    un_masked2 = np.logical_not(mask2) # 1 if finished

    x_rev = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x_rev, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol_rev = Volume[cells]

    #densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens_rev = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    k=0
    k_rev=0

    mask2_rev = dens > ncrit
    un_masked2_rev = np.logical_not(mask2_rev)

    #while np.any(mask2) or np.any(mask2_rev): 
    while np.any(mask2) and (k + 1 < N) or np.any(mask2_rev) and (k_rev + 1 < N):

        mask2_rev = dens_rev > ncrit
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

        mask2 = dens > ncrit # above threshold
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
    
    print("\n", l0, m0, l0*m0)
    print(numb_densities.shape)
    print(radius_vectors.shape)
    print("null means good", np.sum(np.where(numb_densities == 0)), "\n") # if this prints null then we alright

    for x in range(0, l0*m0, l0): 

        mean_columns[i] = np.mean(column_densities[x:x+l0])
        median_columns[i] = np.median(column_densities[x:x+l0])
        i += 1

    if True: # for a single point in space with l0 directions
        fig0, ax0 = plt.subplots()
        h = 0
        ax0.set_xlabel(r'''$s$ [cm]''' + f'''
        x0 = [{Center[0], Center[1], Center[2]}]
        cloud = {h}
        ''', fontsize=16)
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

    print(mean_columns.shape)
    print(median_columns.shape)
    return radius_vectors, numb_densities, mean_columns, median_columns

def density_profile(x_init=np.array([0,0,0]), directions=fibonacci_sphere(), n_crit = threshold):
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
    m0 = 1 # x_init.shape[0]
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

            _vol = np.min(vol)

            dx_vec = ((4 / 3) * _vol / np.pi) ** (1 / 3)

            threshold += mask.astype(int)  # Increment threshold count only for values still above 100

            x += dx_vec * directions

            line[k+1,:,:]    = x
            densities[k+1,:] = dens

            k += 1
        
    threshold = threshold.astype(int)

    max_th     = np.max(threshold) + 1

    #radius_vectors = np.append(line_rev[:max_th_rev,:,:][::-1, :, :], line[1:max_th,:,:], axis=0)*pc_to_cm
    #numb_densities = np.append(densities_rev[:max_th_rev,:][::-1, :], densities[1:max_th,:], axis=0)
    #column_densities = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) # list of 'm' column densities
    #density_profile(x_init=np.array([0,0,0]), directions=fibonacci_sphere(), n_crit = threshold):
    
    mean_columns   = np.zeros(m0)
    median_columns = np.zeros(m0)
    rho_density_mean    = np.zeros(numb_densities.shape[0])
    rho_density_median    = np.zeros(numb_densities.shape[0])
    i = 0

    for x in range(0, l0*m0, l0): 

        rho_density_mean[i] = np.mean(0.5*(densities + densities_rev), axis = 1)
        rho_density_median[i] = np.median(0.5*(densities + densities_rev), axis = 1)

        i += 1
    radius = np.linalg.norm(np.diff(radius_vectors[:,0,:], axis=0), axis=1)

    fig0, ax = plt.subplot()

    ax0.set_xlabel(r'''x_input index''', fontsize=16)
    ax0.set_ylabel(r'$log_{10}(N_{los} /N_{path})$ [adimensional]', fontsize=16)
    ax0.set_yscale('log')
    ax0.set_xscale('log')

    ax.plot(radius, rho_density_mean, '.-',label=r'\rho_{mean}(r)')
    ax.plot(radius, rho_density_median, '.-', label=r'\rho_{mean}(r)')
    ax0.tick_params(axis='both')
    ax0.grid(True, which='both', alpha=0.3)
    ax0.legend(loc="upper left", fontsize=6)

    plt.title(r'$log_{10}(N_{los}/N_{path})$ vs x_input index', fontsize=16)
    fig.tight_layout()
    plt.savefig(f'./rho_r.png',dpi=300)
    plt.close(fig0)    

    return radius_vectors, numb_densities, mean_columns, median_columns

@timing_decorator
def uniform_in_3d(no, rloc=1.0, ncrit=threshold): # modify
    print("points gen...")
    #np.random.seed(0)
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
        # if aux_vector.shape[0] == 0:
        # raise Exception("[error] this regions density is below the threshold")
        distances = np.linalg.norm(aux_vector, axis=1)
        inside_sphere = aux_vector[distances <= rloc]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] * gr_cm3_to_nuclei_cm3 > ncrit
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
        del aux_vector, distances, inside_sphere, valid_mask, valid_points
    rho_vector = np.array(deepcopy(valid_vectors))
    print("points generated")

    if False:                   
        plt.scatter(rho_vector[:, 0], rho_vector[:,1], color = 'skyblue')
        plt.title('PDF $r = R\sqrt[3]{U(0,1)}$')
        plt.ylabel(r'PDF')
        plt.xlabel("$r$ (pc)")
        plt.grid()
        plt.tight_layout()
        plt.savefig('./images/xyz_pre_distro.png', dpi=300)
        plt.close()

    return rho_vector


# python3 stats.py 1000 10 10 ideal 430 TEST seed > R430TST.txt 2> R430TST_error.txt &

if __name__ == '__main__':

    subdirectory = 'ideal_mhd' if case == 'ideal' else 'ambipolar_diffusion'
    
    file_path       = f'./{case}_cloud_trajectory.txt'
    file_list = glob.glob(f"arepo_data/{subdirectory}/*.hdf5")

    _den = []
    _time = []
    with open(file_path, mode='r') as file:

        csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
        next(csv_reader)  # Skip the header row
        print('File opened successfully')
        for row in csv_reader:
            _den += [float(row[8])]
            _time += [float(row[1])]
            if num_file == str(row[0]):
                print("Match found!")
                Center = np.array([float(row[2]),float(row[3]),float(row[4])])
                snap =str(row[0])
                time_value = float(row[1])
                peak_den =  float(row[8])
    if False:
        _den = np.array(_den[::-1])
        _time = np.array(_time[::-1])
        dn_dt = np.gradient(np.log10(_den), _time) #np.diff(_den)/np.diff(_time)
        dt_dstep  = np.gradient(_time)
        # 0.06    Myrs is the maximum dt (600    kyrs)
        # 1.52e-5 Myrs is the minimum dt (0.0152 kyrs)
        # after core density reached 10^6 cm-3, then continous decrease of dt
        fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex = True)

        axes[0,0].scatter(_time, dt_dstep, marker = '.')
        axes[0,0].set_ylabel('$\Delta t$ at snapshot [Myrs]', fontsize=12)
        _text = r'''$\max\{dt\} = 0.06 \rm Myrs = 600 \rm kyrs$
                    $\min\{dt\} = 1.52 \times 10^{-5} \rm Myrs = 0.0152 \rm kyrs$'''
        axes[0,0].text(0.05, 0.075, _text, transform=axes[0,0].transAxes,
                    ha='left', va='bottom', fontsize=6, color='red')
        _time_grad_kyrs = np.gradient(_time*1_000.0)
        ax2 = axes[0,0].twinx()
        ax2.plot(_time, _time_grad_kyrs, '--')
        ax2.set_ylabel('$\Delta t$ [kyrs]', fontsize=12)
        #ax2.set_yscale('log')


        #axes[1,0].plot(_time, dn_dt, '.', color='black')
        axes[1,0].plot(_time, _den, '-', color = 'orange', label=r'$n_{core}$')
        axes[1,0].set_ylabel(r'Density [cm$^{-3}$]', fontsize=14)
        axes[1,0].set_xlabel(r'time [Myrs]', fontsize=14)
        axes[1,0].set_yscale('log')

        from scipy.signal import find_peaks

        peaks, _ = find_peaks(_den)
        axes[1,0].vlines(_time[peaks], ymin=axes[1,0].get_ylim()[0], ymax=np.max(_den), colors='grey', linestyles='dashed', label='Local maxima')
        axes[1,0].set_yscale('log')
        axes[1,0].legend(loc='lower center', fontsize=6)

        ax2 = axes[1,0].twinx()
        ax2.scatter(_time, dt_dstep, marker='.', color = 'green', s=15)
        ax2.plot(_time, dt_dstep, '-', color = 'green', label=r'Num. Derivative') #$\frac{\Delta t}{\Delta (snap)}$
        #ax2.plot(_time, dn_dt, '.-', color='green')
        ax2.set_ylabel('Num. Derivative $\partial_{i} t$', fontsize=14)
        ax2.set_yscale('log')

        lines1, labels1 = axes[1,0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1,0].legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=6)


        axes[0,1].plot(_time, dn_dt, '-', color='red', linewidth=1)
        axes[0,1].set_ylabel(r'$\partial_{t} n_{core}$', fontsize=16)
        axes[0,1].set_yscale('log')
        axes[0,1].tick_params(axis='y', labelcolor='red')
        _text = 'How the peak density grows in time'
        axes[0,1].text(0.5, 0.85, _text, transform=axes[0,1].transAxes,
                    ha='center', va='bottom', fontsize=6, color='black')

        ax2 = axes[0,1].twinx()
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.scatter(_time, _den, marker='.', s=5, color='blue')
        ax2.set_ylabel('$n_g$ [cm$^{-3}$]', fontsize=12)
        ax2.set_yscale('log')

        axes[1,1].tick_params(axis='y', labelcolor='red')
        axes[1,1].plot(_time, dn_dt, '.-', color='red', lw=1, label=r'ND dn/dt')
        axes[1,1].set_ylabel(r'Density [cm$^{-3}$]', fontsize=14)
        axes[1,1].set_xlabel(r'time [Myrs]', fontsize=14)
        axes[1,1].set_yscale('log')

        peaks, _ = find_peaks(_den)
        axes[1,1].vlines(_time[peaks], ymin=axes[1,0].get_ylim()[0], ymax=np.max(_den), colors='gray', linestyles='dashed', lw =1, label='Local maxima')
        axes[1,1].set_yscale('log')
        #axes[1,1].set_xlim(-0.2,4.3)

        ax2 = axes[1,1].twinx()
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.plot(_time, dt_dstep, '.-', color = 'blue', lw=1, label=r'ND dt/d(step)')
        #ax2.scatter(_time, dt_dstep, s=5, color = 'blue') #$\frac{\Delta t}{\Delta (snap)}$
        ax2.set_ylabel('Num. Derivative $\partial_{i} t$', fontsize=14)
        ax2.set_yscale('log')
        lines1, labels1 = axes[1,1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        axes[1,1].legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=6)
        # Add grid to all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.grid(True)

        plt.tight_layout()
        plt.savefig('./timestep.png',dpi=500)
        plt.close(fig)


        # Courant Factor = 0.3 <= adapts timestep base on grid size and velocity
        # 0.06    Myrs is the maximum dt (600    kyrs)
        # 1.52e-5 Myrs is the minimum dt (0.0152 kyrs)
        # after core density reached 10^6 cm-3, then continous decrease of dt
        # Should try to calculate mass to flux ratio or smth similar

        PARAMETERS : dict = {
            'CourantFac': np.float64(0.3),     #  C = u dt/dx <= 0.3 standard value
            'InitGasTemp': np.float64(4500.0), # in Kelvin
            'JeansNumber': np.float64(32.0),   # possibly related to resolution down to the jeans length parameter
            'MaxSizeTimestep': np.float64(0.1),# maximum timestep possible
            'MaxTimebinSpread': np.int32(8),   
            'MaxVolume': np.float64(128.0),    # max cell volume
            'MinSizeTimestep': np.float64(1e-20),# miniimum timestep possible
            'TimeBegin': np.float64(0.0),
            'TimeBetSnapshot': np.float64(5e-07),
            'TimeMax': np.float64(12.8),
            'TimeOfFirstSnapshot': np.float64(0.0),
            'TypeOfTimestepCriterion': np.int32(0),
            'WaitingTimeFactor': np.float64(1.0)}

    if False:
        h = 3
        Center = np.array([106.62494306917117,205.67415032302878,36.807007547204876])
        peak_den = 13939.02078042445

    try:
        print(Center)
    except:
        raise ValueError('Center is not defined')

    filename = None
    for f in file_list:    
        if num_file in f:
            filename = f
            break
    if filename is None:
        raise FileNotFoundError

    os.makedirs("stats", exist_ok=True)
    parent_folder = "stats/"+ case 
    children_folder = os.path.join(parent_folder, snap)
    os.makedirs(children_folder, exist_ok=True)

    data = h5py.File(filename, 'r')
    Boxsize = data['Header'].attrs['BoxSize']
    Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
    #MagneticFieldDivergence = np.asarray(data['PartType0']['MagneticFieldDivergence'], dtype=FloatType)
    #MagneticFieldDivergenceAlt = np.asarray(data['PartType0']['MagneticFieldDivergenceAlternative'], dtype=FloatType)
    Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
    VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
    #Velocity = np.asarray(data['PartType0']['Velocities'], dtype=FloatType)
    Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
    Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
    Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
    Bfield_grad = np.zeros((len(Pos), 9))
    Density_grad = np.zeros((len(Density), 3))
    Volume   = Mass/Density
    # -------------------------
    #Center = VoronoiPos[np.argmax(Density), :]
    # -------------------------
    VoronoiPos-=Center
    Pos-=Center
    """
    for dim in range(3):  # Loop over x, y, z
        pos_from_center = Pos[:, dim]
        boundary_mask = pos_from_center > Boxsize / 2
        Pos[boundary_mask, dim] -= Boxsize
        VoronoiPos[boundary_mask, dim] -= Boxsize

    """
    for dim in range(3):  # Loop over x, y, z
        pos_from_center = Pos[:, dim]
        too_high = pos_from_center > Boxsize / 2
        too_low  = pos_from_center < -Boxsize / 2
        Pos[too_high, dim] -= Boxsize
        Pos[too_low,  dim] += Boxsize

    print("Cores Used          : ", os.cpu_count())
    print("Steps in Simulation : ", 2*N)
    print("rloc                : ", rloc)
    print("max_cycles          : ", max_cycles)
    print("Boxsize             : ", Boxsize) # 256
    print("Center              : ", Center) # 256
    print("Smallest Volume     : ", Volume[np.argmin(Volume)]) # 256
    print("Biggest  Volume     : ", Volume[np.argmax(Volume)]) # 256
    print(f"Smallest Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmax(Volume)]}")
    print(f"Biggest  Density (N/cm-3)  : {gr_cm3_to_nuclei_cm3*Density[np.argmin(Volume)]}")
    print("Allocation Number: ", 2*N)

    #x_input = np.vstack([uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud), np.array([0.0,0.0,0.0])])
    x_input = uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud)
    directions=fibonacci_sphere(20)

    #density_profile(x_input, directions=fibonacci_sphere(100), n_crit = threshold)

    radius_vectors_los, numb_densities_los, mean_column, median_column                     = line_of_sight(x_input, directions, threshold)
    radius_vectors, magnetic_fields, numb_densities, follow_index, path_column, survivors1 = crs_path(x_input, threshold)

    #data.close()

    if np.any(numb_densities > threshold):
        r_u, n_rs, B_rs, survivors2 = eval_reduction(magnetic_fields, numb_densities, follow_index, threshold)
        r_l, _1, _2, _3 = eval_reduction(magnetic_fields, numb_densities, follow_index, threshold) # not needed
        print("DEBUG numb_densities type:", type(numb_densities))
    else:
        print(f"[error] No densities above {threshold} cm⁻³")
        raise ValueError

    print(np.sum(survivors1), survivors1.shape)
    print(np.sum(survivors2), survivors2.shape) 

    survivors = np.logical_and(survivors1, survivors2)

    print(f"Elapsed Time: ", (time.time()-start_time)//60., " Minutes")

    print(np.sum(survivors))  

    # survivors is a mask array that discriminates field lines longer that the allocated size. This are usually small > 5 %
    u_input         = x_input[np.logical_not(survivors),:] 
    x_input         = x_input[survivors,:]
    radius_vectors  = radius_vectors[:, survivors, :]
    numb_densities  = numb_densities[:, survivors]
    magnetic_fields = magnetic_fields[:, survivors]
    mean_column     = mean_column[survivors]
    median_column   = median_column[survivors]
    path_column     = path_column[survivors]
    n_rs = n_rs[survivors]
    B_rs = B_rs[survivors] 
    r_u  = r_u[survivors]
    r_l  = r_l[survivors]

    ratio_mean = np.log10(mean_column/path_column)
    ratio_median = np.log10(median_column/path_column)

    if False:
        for p, _ in enumerate(ratio_mean):
            
            if not (abs(_) > 2):
                #print(f"[info] {p} - N_los/N_path < 10^2 for all lines")
                continue

            print(path_column[p], ratio_mean[p], ratio_median[p])
            fig0, ax0 = plt.subplots()
            h = 0
            l0 = directions.shape[0]
            ax0.set_xlabel(r'''x_input index''' + f'''
            x0 = [{Center[0], Center[1], Center[2]}]
            cloud = {h}, point {x_input[p]}
            ''', fontsize=16)
            ax0.set_ylabel(r'$log_{10}(N_{los} /N_{path})$ [adimensional]', fontsize=16)
            #ax0.set_yscale('log')
            ax0.set_xscale('log')

            ax0.scatter(path_column, ratio_mean, color='blue', s=5)
            ax0.scatter(path_column, ratio_median, color='green', s=5)

            ax0.scatter(path_column[p], ratio_mean[p], label=f'{p}-mean-to-index', color='blue', s=5)
            ax0.scatter(path_column[p], ratio_median[p], label=f'{p}-median-to-index', color='green', s=5)

            ax0.tick_params(axis='both')
            ax0.grid(True, which='both', alpha=0.3)
            ax0.legend(loc="upper left", fontsize=6)

            plt.title(r'$log_{10}(N_{los}/N_{path})$ vs x_input index', fontsize=16)
            fig0.tight_layout()
            plt.savefig(f'./ratio_vs_los_{h}-{p}.png',dpi=300)
            plt.close(fig0)

    distance = np.linalg.norm(x_input, axis=1)*pc_to_cm

    # cr path column densities
    path_column   = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) *pc_to_cm

    # free space
    del data, Mass, Density_grad, Bfield_grad

    print((time.time()-start_time)//60, " Minutes")

    # size of arrays
    m = magnetic_fields.shape[1]
    N = magnetic_fields.shape[0]

    print(magnetic_fields.shape)

    # reduction factor
    subunity = r_u < 1
    unity    = r_u == 1
    rho_subunity = x_input[subunity,:]
    rho_unity = x_input[unity,:]

    # removing PATH/LOS passing through core
    order    = np.argsort(distance)
    path_column_ordered = path_column[order][1:]
    los_column_ordered = mean_column[order][1:]
    s_ordered = distance[order][1:]
    los_column  = mean_column.copy()
    s      = distance.copy()

    # retio
    ratio = path_column/los_column

    if True: # path_column & los_column Linear fit vs. Distance away from core
        fig, ax = plt.subplots()
        ax.set_xlabel(r'''$N_{path}$ (cm$^{-2}$)''', fontsize=16)
        ax.set_ylabel(r'$N_{los}$ (cm$^{-2}$)', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.scatter(path_column, los_column, marker='^',color="black", s=5)

        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)
        min_ = min(np.min(los_column), np.min(path_column))
        max_ = min(np.max(los_column), np.max(path_column))
        ax.set_ylim(min_, max_)
        ax.set_xlim(min_, max_)
        #ax.legend(loc="upper left", fontsize=12)
        
        plt.title(r"Correlation between columns densities", fontsize=16)
        fig.tight_layout()
        plt.savefig('./images/columns/column_column.png', dpi=300)
        plt.close()

    if True: # path_column & los_column Linear fit vs. Distance away from core
        fig, ax = plt.subplots()

        log_y = np.log10(los_column_ordered); m1, b1 = np.polyfit(s_ordered, log_y, 1); fit1 = 10**(m1 * s_ordered + b1)
        log_y2 = np.log10(path_column_ordered); m2, b2 = np.polyfit(s_ordered, log_y2, 1); fit2 = 10**(m2 * s_ordered+ b2)

        eq1 = rf"$\log_{{10}}(N_{{los}}) = {m1:.4e}\,s + {b1:.4f}$"
        eq2 = rf"$\log_{{10}}(N_{{path}}) = {m2:.4e}\,s + {b2:.4f}$"

        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
        ax.set_ylabel(r'$N$ (cm$^{-2}$)', fontsize=16)
        ax.set_yscale('log')
        #ax.set_xscale('log')

        ax.scatter(s_ordered, los_column_ordered, marker='o',color="#8E2BAF", s=5, label=r'$N_{\rm los}$')
        ax.scatter(s_ordered, path_column_ordered, marker ='v',color="#148A02", s=5, label=r'$N_{\rm path}$')
        ax.plot(s_ordered, fit1, '-' , color="black", linewidth=1,label='$N_{los} fit$')
        ax.plot(s_ordered, fit2, '--', color="black", linewidth=1,label='$N_{path} fit$')

        ax.text(0.05, 0.65, eq1, transform=ax.transAxes, color="#8E2BAF", fontsize=12, va="top")
        ax.text(0.05, 0.60, eq2, transform=ax.transAxes, color="#148A02", fontsize=12, va="top")

        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc="upper left", fontsize=12)
        
        plt.title(r"Ratio Column Density (path/los)", fontsize=16)
        fig.tight_layout()
        plt.savefig('./images/columns/column_comparison.png', dpi=300)
        plt.close()

    if True:
        distance = np.linalg.norm(x_input, axis=1)*pc_to_cm

        high_ratio = ratio >= 1.0e+1
        low_ratio  = ratio <  1.0e+1

        print(distance.shape)
        print(ratio.shape)
        print(high_ratio.shape)
        print(low_ratio.shape)

        fig, ax = plt.subplots()

        #ax.scatter(s, ratio, marker='x', color="black", s=5, alpha=0.3)
        ax.scatter(s[high_ratio], ratio[high_ratio], marker='o', color="#8E2BAF", s=5, label=r'$\log_{10}(\frac{N_{los}}{N_{los}}) \geq 1$')
        ax.scatter(s[low_ratio],ratio[low_ratio], marker='v', color="#148A02", s=5, label=r'$\log_{10}(\frac{N_{los}}{N_{los}}) < 1$')
        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
        ax.set_ylabel(r'$ratio$', fontsize=16)
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.legend(loc="upper left", fontsize=12)    
        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()        
        plt.xlabel('Distance to origin [pc]')
        plt.ylabel(r'$\log_{10}(\frac{N_{los}}{N_{los}}) \geq 2$')
        plt.title('Density Map of Astrophysical System (Irregular Data)')
        plt.savefig('./images/ratio_columns1.png', dpi=300)
        plt.close(fig)

    R = 0.0025 # Radius 5000 [Au]
    H = 1.0e-2 # Height [Pc]

    x = Pos[:, 0] 
    y = Pos[:, 1] 
    z = Pos[:, 2] 
    c = (x**2 + y**2 + z**2 < R**2)
    s = np.logical_and(-H < z, z < H)
    d = Density * gr_cm3_to_nuclei_cm3 > 1.0e+2
    csd = np.logical_and(np.logical_and(c, s), d)
    

    if True: # Density Hexbin heatmap
        
        c_Density = Density[csd] * gr_cm3_to_nuclei_cm3
        xc = Pos[csd, 0] 
        yc = Pos[csd, 1] 
        zc = Pos[csd, 2] 
        
        fig, ax = plt.subplots()

        plt.figure(figsize=(8, 6))

        hb = plt.hexbin(xc, yc, C=c_Density, gridsize=70, cmap='inferno', norm=LogNorm())        
        cb = plt.colorbar(hb, label='Density')  # Colorbar to show the density scale
        ax.legend(loc="upper left", fontsize=12)    
        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)

        #x_start = 0
        #y_start = R*0.5
        #bar_length = R*0.5
        #ax.hlines(y_start, x_start, x_start + bar_length, colors='black', linewidth=3)
        #ax.text(x_start + bar_length/2, y_start - 0.5, f'{R/2} Pc', ha='center')

        plt.xlabel('X Position (Au)')
        plt.ylabel('Y Position (kpc)')
        plt.title('Density Map of Astrophysical System (Irregular Data)')
        plt.savefig('./images/density_map.png', dpi=300)
        plt.close(fig)
        
    if True: # Streamplot
        x = Pos[:, 0] 
        y = Pos[:, 1] 
        z = Pos[:, 2] 

        c_Density = Density[csd] * gr_cm3_to_nuclei_cm3

        X = Pos[csd, 0] 
        Y = Pos[csd, 1]
        print(X.shape, Y.shape)
        vec = Pos[csd,:]

        dist, cells, rel_pos = find_points_and_relative_positions(Pos[csd,:], Pos, VoronoiPos)
        local_fields         = get_magnetic_field_at_points(Pos[csd,:], Bfield, rel_pos)
        
        print(local_fields.shape)

        __step__ = local_fields.shape[0] # if I want 

        U = local_fields[:,0]
        V = local_fields[:,1]
        #X = X
        #Y = Y
        D = c_Density
        
        print(U.shape, V.shape, D.shape)
        # Normalize vector length and color by magnitude
        __mag__ = np.linalg.norm(np.array([U, V]), axis=0)

        # Avoid division by zero
        U_norm = U / (__mag__ + 1e-20)
        V_norm = V / (__mag__ + 1e-20)

        # Normalize vector length and color by magnitude
        __mag__ = np.linalg.norm(np.array([U, V]), axis=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot density background
        contour = ax.tricontourf(X, Y, __mag__, levels=15, cmap='viridis')

        # Quiver with color corresponding to vector magnitude
        Q = ax.quiver(X, Y, U_norm, V_norm, __mag__, cmap='plasma', scale=30, width=0.003, alpha=0.8, norm=LogNorm())
        #Strm = ax.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')

        #fig.colorbar(Strm.lines)
        # Add colorbar for magnitude
        cbar = plt.colorbar(Q, ax=ax, label='|B| (µG)', pad=0.02)
        cbar.ax.set_ylabel('|B| (µG)', rotation=270, labelpad=15)

        # Labels and layout
        ax.set_xlabel('X Position (kpc)')
        ax.set_ylabel('Y Position (kpc)')
        ax.set_title('Magnetic Field and Density Map (Irregular Data)')
        ax.set_aspect('equal', 'box')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.02, 0.95, '1', transform=ax.transAxes, fontsize=18, color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig('./images/streamplot.png', dpi=300)
        plt.close(fig)
    if True: # Hexbin R
        pass

    if True: # path_column & los_column vs. Density
        log_x = np.log10(n_rs)
        log_y = np.log10(los_column); m1, b1 = np.polyfit(log_x, log_y, 1); fit1 = 10**(m1 * log_x + b1)
        log_y2 = np.log10(path_column); m2, b2 = np.polyfit(log_x, log_y2, 1); fit2 = 10**(m2 * log_x+ b2)
        
        fig, ax = plt.subplots()

        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax.set_ylabel(r'$N$ (cm$^{-2}$)', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')

        eq1 = rf"$\log_{{10}}(N_{{los}}) = {m1:.5f}\,\log_{{10}}(n_g) + {b1:.5f}$"
        eq2 = rf"$\log_{{10}}(N_{{path}}) = {m2:.5f}\,\log_{{10}}(n_g) + {b2:.5f}$"
        ax.text(0.05, 0.85, eq1, transform=ax.transAxes, color="#8E2BAF", fontsize=12, va="top")
        ax.text(0.05, 0.80, eq2, transform=ax.transAxes, color="#148A02", fontsize=12, va="top")

        ax.plot(n_rs, fit1, '-', color="black", linewidth=1,label='$N_{los}$')
        ax.plot(n_rs, fit2, '--', color="black", linewidth=1,label='$N_{path}$')

        ax.scatter(n_rs, los_column, marker='o',color="#8E2BAF", s=5, label=r'$N_{\rm los}$', alpha=0.3)
        ax.scatter(n_rs, path_column, marker ='v',color="#148A02", s=5, label=r'$N_{\rm path}$', alpha=0.3)
        
        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)

        plt.title(r"$N(n_g)$", fontsize=16)
        ax.legend(loc="upper right", fontsize=12)
        
        fig.tight_layout()
        plt.savefig('./images/columns/column_density.png', dpi=300)
        plt.close()

    log_ionization_path_l, log_ionization_path_h = ionization_rate_fit(path_column)
    log_ionization_los_l, log_ionization_los_h  = ionization_rate_fit(mean_column)

    if True: # Ionization rate (L) vs. Density

        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)    
        ax.set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
        
        #ax.set_yscale('log')
        ax.set_xscale('log')
        
        ax.scatter(n_rs, log_ionization_los_l, marker='o',color="#8E2BAF", s=5, alpha = 0.3, label=r'$\zeta_{\rm los}$')
        ax.scatter(n_rs, log_ionization_path_l, marker ='v',color="#148A02", s=5, alpha = 0.3, label=r'$\zeta_{\rm path}$')

        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=16)
        ax.set_ylim(-19.5, -15.5)

        fig.tight_layout()
        plt.savefig(f'./images/columns/zeta_density_l.png', dpi=300)
        plt.close()

    if True: # Ionization rate (H) vs. Density

        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax.set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
        
        ax.set_xscale('log')
        
        ax.scatter(n_rs, log_ionization_los_h, marker='o',color="#8E2BAF", s=5, alpha = 0.3, label=r'$\zeta_{\rm los}$')
        ax.scatter(n_rs, log_ionization_path_h, marker ='v',color="#148A02", s=5, alpha = 0.3, label=r'$\zeta_{\rm path}$')

        ax.grid(True, which='both', alpha=0.3)
        plt.title(rf"$\zeta(n_g)$ ({case + num_file})", fontsize=16)
        ax.legend(fontsize=16)
        ax.set_ylim(-19.5, -15.5)
        fig.tight_layout()
        plt.savefig(f'./images/columns/zeta_density_h.png', dpi=300)
        plt.close()

    if True:

        fig, (ax_l, ax_h) = plt.subplots(1, 2, figsize=(10, 5),gridspec_kw={'wspace': 0, 'hspace': 0}, sharey=True)

        ax_l.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax_l.set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
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
        plt.savefig('./images/zeta_density_combined.png', dpi=300)
        plt.close()

    ratio_ionization_path_to_los_l = log_ionization_path_l-log_ionization_los_l
    ratio_ionization_path_to_los_h = log_ionization_path_l-log_ionization_los_h

    if True: # Ratio Ionization_path/Ionization_los vs. Density
        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax.set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
        ax.set_xscale('log')
        
        ax.scatter(n_rs, ratio_ionization_path_to_los_l, marker='o',color="#CB71EA", s=5, alpha = 0.3, label='Model $\mathcal{L}$')
        ax.scatter(n_rs, ratio_ionization_path_to_los_h, marker ='v',color="#60D24F", s=5, alpha = 0.3, label='Model $\mathcal{H}$')

        ax.grid(True, which='both', alpha=0.3)
        
        plt.title(rf"$\zeta_{{path}}/\zeta_{{los}} $ ({case + num_file})", fontsize=16)
        ax.set_ylim(-5, 5)

        ax.legend(fontsize=16)
        fig.tight_layout()
        plt.savefig(f'./images/columns/ratio_zeta_density.png', dpi=300)
        plt.close()

    if True: # Ratio Ionization_path & Ionization_los vs. Distance
        fig, ax = plt.subplots()
        
        ax.scatter(distance, ratio_ionization_path_to_los_l, marker='o', color="#CB71EA", s=5, alpha = 0.5, label='Model $\mathcal{L}$')
        ax.scatter(distance, ratio_ionization_path_to_los_h, marker ='v', color="#60D24F", s=5, alpha = 0.5, label='Model $\mathcal{H}$')

        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)    
        ax.set_ylabel(r'Ratio', fontsize=16)
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim(-5, 5)
        plt.title(rf"$\zeta_{{path}}/\zeta_{{los}} $ ({case + num_file})", fontsize=16)
        ax.legend(fontsize=12)
        fig.tight_layout()
        plt.savefig('./images/columns/ratio_zeta_distance.png', dpi=300)
        plt.close()

    if True: # Reduction factor vs. R -- (R < 1) Scatter and Histogram
        s_subunity = distance[subunity]
        r_u_subunity = r_u[subunity]
        
        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
        ax.set_ylabel(r'$1-R$', fontsize=16)
        #ax.set_yscale('log')
        ax.set_xscale('log')

        ax.scatter(s_subunity, r_u_subunity, marker ='o',color='black', s=5, label="$R<1$")

        ax.grid(True, which='both', alpha=0.3)
        
        plt.title(rf"$1 - R(s)$  ({case + num_file})", fontsize=16)
        ax.legend(fontsize=12)
        
        fig.tight_layout()
        plt.savefig('./images/reduction/r_subunity_complement.png', dpi=300)
        plt.close()

        # 1 - R Histogram vs. distance
        fig, ax = plt.subplots(figsize=(8, 3.5))

        bins = np.linspace(
            np.min(1 - r_u_subunity),
            np.max(1 - r_u_subunity),
            r_u_subunity.shape[0] // 20
        )

        ax.hist(1 - r_u_subunity, bins=bins, alpha=1,
                histtype='stepfilled', label=f"{time} Myrs")

        print("Mean R<1: ", np.mean((1 - r_u_subunity)))

        ax.set_yscale('log')
        ax.set_ylabel('Counts', fontsize=16)
        ax.set_xlabel('''$\log_{10}(\zeta / s^{-1})$''', fontsize=12)
        ax.legend(fontsize=12)

        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./images/reduction/r_complement_histogram.png', dpi=300)

    if False: # Note to self, for small rloc < 1 this code doesnt take long
        # Subunity
        fig, ax = plt.subplots()
        ax.scatter(rho_subunity[:, 0], rho_subunity[:, 1], color='red', s=8, alpha=0.3, label="$R<1$")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig('./images/xyz_distro/xy_subunity.png', dpi=300)
        plt.close(fig)

        # Unity
        fig, ax = plt.subplots()
        ax.scatter(rho_unity[:, 0], rho_unity[:, 1], color='red', s=8, alpha=0.3, label="$R=1$")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig('./images/xyz_distro/xy_unity.png', dpi=300)
        plt.close(fig)

        # Both
        fig, ax = plt.subplots()
        ax.scatter(rho_unity[:, 0], rho_unity[:, 1], color='red', s=8, alpha=0.3, label="$R=1$")
        ax.scatter(rho_subunity[:, 0], rho_subunity[:, 1], color='black', s=8, alpha=0.3, label="$R<1$")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig('./images/xyz_distro/xy_full.png', dpi=300)
        plt.close(fig)

    reduction_density(r_u, n_rs, 'u')#; reduction_density(r_l, n_rs, 'l')

    if False:
        r_u = np.array(r_u) # threshold * 1
        r_l = np.array(r_l) # threshold * 10

        total = r_u.shape[0]
        mask = r_u == 1.0  
        ones = np.sum(mask)
        fraction = ones/total
        x_ones = x_input[mask,:]
        x_not  = x_input[np.logical_not(mask),:]

        tda(x_input, 'x')
        tda(x_ones, 'R = 1')
        tda(x_not , 'R < 1')


    m = radius_vectors.shape[1]

    zoom = 2*rloc#np.mean(np.sqrt(x**2 + y**2 + z**2))-2

    if False:
            
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

    if False:
        try:
                
            from matplotlib import cm
            from matplotlib.colors import Normalize



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

                norm = Normalize(vmin=np.min(magnetic_fields[:, k]), vmax=np.max(magnetic_fields[:,k]))
                cmap = cm.viridis

                ax.scatter(x0, y0, z0, marker="x",color="black",s=3, alpha=0.5)            
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


    #h5_path = os.path.join(children_folder, f"DataBundle.h5")

    #from filelock import FileLock
    """
    Time ──────────────────────────────────────────▶

    Script 1:  ──[Acquire Lock]───[Create/Append HDF5]───[Release Lock]───

    Script 2:  ──────────[Wait for Lock]───────────────────────────[Append HDF5]───[Release Lock]───
    """
    #lock = FileLock(h5_path + ".lock")

    #with lock:  # acquire lock (waits if another process is holding it)

    #    pass
    
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    simulation_name = "Magnetic Pockets"
    runtime =(time.time()-start_time)//60


    pdf_file = "stats/single_stats.pdf"
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
        f"Assumptions:        {case}",
        f"Cloud Density:      {dense_cloud}",
        f"Threshold Density:  {threshold}",
        f"Filtered Lines:     {np.sum(np.logical_not(survivors))}",
        f"Final Lines:        {np.sum(survivors)}",
        f"Fraction Non-CNVRG Lines:  {np.sum(np.logical_not(survivors))/np.sum(survivors)}",
    ]

    y = height - 120
    for line in lines:
        c.drawString(72, y, line)
        y -= 20

    # Add PNG image
    image_file = "images/FieldTopology.png"  # replace with your PNG path
    img_width = 400  # width in points
    img_height = 300  # height in points
    y_image = y - img_height - 20  # position below the last line of text

    c.drawImage(image_file, 72, y_image, width=img_width, height=img_height)

    c.save()
    print(f"Report saved to {pdf_file}")