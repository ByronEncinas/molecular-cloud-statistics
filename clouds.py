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
    how_many   = int(sys.argv[3])
except:
    input_case = 'amb'
    how_many   = 5

if input_case == 'ideal':
    file_list = [sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5'))]
elif input_case == 'amb':
    file_list = [sorted(glob.glob('arepo_data/ambipolar_diffusion/*.hdf5'))]


for list in file_list:
    for fileno, filename in enumerate(list[::-1]):
        data = h5py.File(filename, 'r')
        header_group = data['Header']
        time_value = header_group.attrs['Time']
        snap = filename.split('/')[-1].split('.')[0][-3:]
        parent_folder = "clouds/"+input_case
        children_folder = os.path.join(parent_folder, snap)
        os.makedirs(children_folder, exist_ok=True)
        Boxsize = data['Header'].attrs['BoxSize']
        #VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
        Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
        Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
        #Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
        
        xc = Pos[:, 0]
        yc = Pos[:, 1]
        zc = Pos[:, 2]

        region_radius = 10

        ds = yt.load(filename)

        centers = np.zeros((how_many,3))

        for each in range(how_many):

            if each == 0:
                CloudCord = Pos[np.argmax(Density), :]
                cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)
                Density[cloud_sphere] *= 0.0
                PeakDensity = np.log10(Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3)
                with open(f"clouds/{input_case}/{snap}/clouds.txt", "w") as file:
                    file.write("snap,index,CloudCord_X,CloudCord_Y,CloudCord_Z,Peak_Density\n")
                    file.write(f"{snap},{each},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")
                Density[cloud_sphere] = 0.0

                sp = yt.SlicePlot(
                    ds, 
                    'y', 
                    ('gas', 'density'), 
                    center=[CloudCord[0], CloudCord[1], CloudCord[2]],
                    width=256* u.pc
                )
                sp.annotate_sphere(
                    [CloudCord[0], CloudCord[1], CloudCord[2]],  # Center of the sphere
                    radius=(region_radius*0.5, "pc"),  # Radius of the sphere (in physical units, e.g., pc)
                    circle_args={"color": "black", "linewidth": 2}  # Styling for the sphere
                )
                centers[each,:] = [CloudCord[0], CloudCord[1], CloudCord[2]]
                
            else:
                cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)
                Radius = np.sqrt((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2)
                Density[cloud_sphere] = 0.0

                CloudCord = Pos[np.argmax(Density), :]
                PeakDensity = np.log10(Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3)

                with open(f"clouds/{input_case}/{snap}/clouds.txt", "a") as file:
                    file.write(f"{snap},{each},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")

                sp.annotate_sphere(
                    [CloudCord[0], CloudCord[1], CloudCord[2]],  # Center of the sphere
                    radius=(region_radius*0.5, "pc"),  # Radius of the sphere (in physical units, e.g., pc)
                    circle_args={"color": "black", "linewidth": 2}  # Styling for the sphere
                )
                centers[each,:] = [CloudCord[0], CloudCord[1], CloudCord[2]]

                sp.annotate_text(
                    [CloudCord[0]+8, CloudCord[1]+8, CloudCord[2]+8],  # Slightly offset to avoid overlap
                    text=f"{each}",  # Your custom label
                    coord_system="data",  # Use data coordinates
                    text_args={"color": "black", "fontsize": 12}
                )

        print(centers)

        sp.annotate_timestamp(redshift=False)

        sp.save(os.path.join(parent_folder, f"{input_case}_{snap}.png"))

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
threshold = 1.0e+1
N = 30

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
            axs.set_yscale('log')
            axs.set_xlabel("Index")
            axs.set_ylabel(label)
            axs.set_title(f"{label} Shape")
            axs.legend()
            axs.grid(True)

        # Plot both subplots
        #plot_field(axs_numb, numb, "Density")
        #plot_field(axs_bfield, bfield, "Magnetic Field")

        #plt.tight_layout()
        #plt.savefig('./images/columns/mosaic.png')
        #plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def evaluate_reduction(field, numb, follow_index):

    R10      = []
    R100     = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    print("_, m = field.shape => ", _, m)
    
    for i in range(m):

        #_100 = np.where(numb[:, i] > 1.0e+2)[0]
        #_10  = np.where(numb[:, i] > 1.0e+1)[0]
        # must be k + k_rev + 1 in size
            
        # this must contain list of indexes of positions with densities > 10cm-3
        # this should be empty for at least one
        mask10 = np.where(numb[:, i] > 1.0e+1)[0]

        if mask10.size > 0:
            start, end = mask10[0], mask10[-1]
            #print(mask10)
            print(i, start, end)

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
        mask100 = np.where(numb[:, i] > threshold)[0]
        
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

def eval_reduction(field, numb, follow_index, threshold):

    R10      = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    #print("_, m = field.shape => ", _, m)
    from collections import Counter
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
        
        # Flatten and count
        counter = Counter(bfield10.ravel())  # ravel() flattens the array
        most_common_value, count = counter.most_common(1)[0]


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

        if count > 20:
            # if count > 20 we assume that there is a jump into low resolution in magnetic field
            # which happens at low densities, so we constrict field lines by this 
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)   
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

        if count > 10:
            flag = True
            R10[-1] = 1.0
            print(f"Most common value: {most_common_value} (appears {count} times): R = ", R)

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

            #print(max(np.max(np.linalg.norm(line[k, :, :], axis=1)),np.max(np.linalg.norm(line_rev[k_rev, :, :], axis=1))))

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
        if False:                   
            bins = 100
            plt.hist(r, bins=bins, color = 'skyblue', density =True)
            plt.title('PDF $r = R\sqrt[3]{U(0,1)}$')
            plt.ylabel(r'PDF')
            plt.xlabel("$r$ (pc)")
            plt.grid()
            plt.tight_layout()
            plt.savefig('./images/columns/pdf_r.png')
            plt.close()

            plt.hist(theta, bins=bins, color = 'skyblue', density =True)
            plt.title('PDF $\\theta = \\arccos(2U-1)$')
            plt.ylabel(r'PDF')
            plt.xlabel('$\\theta$ (rad)')
            plt.grid()
            plt.tight_layout()
            plt.savefig('./images/columns/pdf_theta.png')
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

""" From this point on, we can loop through the clouds calculating the column densities"""

max_cycles = 100
rloc = 0.1

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

        Pos+=centers[each, :]
        VoronoiPos-=centers[each, :]

        #x_input = np.vstack([uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud), np.array([0.0,0.0,0.0])])
        x_input = uniform_in_3d(max_cycles, rloc, ncrit=dense_cloud)
        directions = fibonacci_sphere(20)

        radius_vector, numb_densities, average_column = line_of_sight(x_input, directions, threshold)

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
        c_rs   = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vector, axis=0), axis=2), axis=0) *pc_to_cm

        # free space
        data.close()
        del VoronoiPos, Pos, Bfield, Density, Mass, Bfield_grad, Density_grad, Volume
        import gc
        gc.collect()

        print((time.time()-start_time)//60, " Minutes")

        distance = np.linalg.norm(x_input, axis=1)*pc_to_cm
        order = np.argsort(distance)


        N_path = c_rs[order][1:]
        N_los = average_column[order][1:]
        s = distance[order][1:]

        if True:
            fig, ax = plt.subplots()
            
            # Axis labels
            ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
            
            ax.set_ylabel(r'$N$ (cm$^{-2}$)', fontsize=16)
            
            # Log scales
            ax.set_yscale('log')
            #ax.set_xscale('log')
            
            ax.scatter(s, N_los, marker='o',color="#8E2BAF", s=5, label=r'$N_{\rm LOS}$')
            ax.scatter(s, N_path, marker ='v',color="#148A02", s=5, label=r'$N_{\rm PATH}$')
            
            # Fit for N_LOS
            log_y = np.log10(N_los)
            m1, b1 = np.polyfit(s, log_y, 1)
            fit1 = 10**(m1 * s + b1)

            # Fit for N_PATH
            log_y2 = np.log10(N_path)
            m2, b2 = np.polyfit(s, log_y2, 1)
            fit2 = 10**(m2 * s+ b2)

            # Plot fits
            ax.plot(s, fit1, '--', color="#8E2BAF", linewidth=1)
            ax.plot(s, fit2, '--', color="#148A02", linewidth=1)

            # Add fit equations as text (log10 form)
            eq1 = rf"$\log_{{10}}(N_{{LOS}}) = {m1:.4e}\,\log_{{10}}(x) + {b1:.4f}$"
            eq2 = rf"$\log_{{10}}(N_{{PATH}}) = {m2:.4e}\,\log_{{10}}(x) + {b2:.4f}$"

            print(eq1)
            print(eq2)
            # Place them on the plot
            ax.text(0.05, 0.95, eq1, transform=ax.transAxes, color="#8E2BAF", fontsize=16, va="top")
            ax.text(0.05, 0.90, eq2, transform=ax.transAxes, color="#148A02", fontsize=16, va="top")

            # Ticks and grid
            ax.tick_params(axis='both')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Title and legend
            #plt.title(r"$N \propto r$", fontsize=16)
            ax.legend(loc="lower left")
            #ax.legend()

            # Layout and save
            fig.tight_layout()
            
            plt.savefig(os.path.join(children_folder, f'{how_many}clouds_columns.png'), dpi=300)
            plt.close()

        N_path = c_rs[1:]
        N_los = average_column[1:]

        if True:
            fig, ax = plt.subplots()
            
            # Axis labels
            ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
            
            ax.set_ylabel(r'$N$ (cm$^{-2}$)', fontsize=16)
            
            # Log scales
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            # Scatter plot with label for legend
            ax.scatter(n_rs[0], average_column[0], marker='x',color="#8E2BAF", s=8)
            ax.scatter(n_rs[0], c_rs[0], marker ='x',color="#148A02", s=8)
            ax.scatter(n_rs[1:], N_los, marker='o',color="#8E2BAF", s=5, label=r'$N_{\rm LOS}$')
            ax.scatter(n_rs[1:], N_path, marker ='v',color="#148A02", s=5, label=r'$N_{\rm PATH}$')
            
            # Fit for N_LOS
            log_y = np.log10(N_los)
            log_x = np.log10(n_rs[1:])
            m1, b1 = np.polyfit(log_x, log_y, 1)
            fit1 = 10**(m1 * log_x + b1)

            # Fit for N_PATH
            log_y2 = np.log10(N_path)
            m2, b2 = np.polyfit(log_x, log_y2, 1)
            fit2 = 10**(m2 * log_x+ b2)

            # Plot fits
            ax.plot(n_rs[1:], fit1, '--', color="#8E2BAF", linewidth=1)
            ax.plot(n_rs[1:], fit2, '--', color="#148A02", linewidth=1)

            # Add fit equations as text (log10 form)
            eq1 = rf"$\log_{{10}}(N_{{LOS}}) = {m1:.5f}\,\log_{10}(x) + {b1:.5f}$"
            eq2 = rf"$\log_{{10}}(N_{{PATH}}) = {m2:.5f}\,\log_{10}(x) + {b2:.5f}$"
            ax.text(0.05, 0.95, eq1, transform=ax.transAxes, color="#8E2BAF", fontsize=16, va="top")
            ax.text(0.05, 0.90, eq2, transform=ax.transAxes, color="#148A02", fontsize=16, va="top")

            # Ticks and grid
            ax.tick_params(axis='both')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Title and legend
            plt.title(r"$N \propto n_g$", fontsize=16)
            ax.legend(loc="upper right")
            
            # Layout and save
            fig.tight_layout()
            plt.savefig(os.path.join(children_folder, f'{how_many}clouds_ion.png'), dpi=300)
            plt.close()
