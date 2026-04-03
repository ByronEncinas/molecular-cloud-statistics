import csv, glob, os, sys, time, h5py
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from copy import deepcopy
from scipy.spatial import cKDTree
import warnings
from src.library import *
import pandas as pd
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
MARKERS = ['v', 'o']
COLORS  = ["#8E2BAF", "#148A02"]
ALPHA   = 0.9
SIZE    = 8
FONTSIZE = 18
GRID_ALPHA = 0.5

start_time = time.time()

cases = ['ideal', 'amb']

s_ideal = []
s_amb = []
t_ideal = []
t_amb = []
nc_ideal = []
nc_amb = []

for case in cases:
    file_path = f'./util/{case}_cloud_trajectory.txt'
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if case == "ideal":
                s_ideal.append(str(row[0]))
                t_ideal.append(float(row[1]))
                nc_ideal.append(float(row[-1]))

            if case == "amb":
                s_amb.append(str(row[0]))
                t_amb.append(float(row[1]))
                nc_amb.append(float(row[-1]))

mu_mH = 2.35 * 1.67e-24
rho_ideal = np.array(nc_ideal) * mu_mH
rho_amb = np.array(nc_amb) * mu_mH

_min, _max = min(rho_ideal.min(), rho_amb.min())*1.1, max(rho_ideal.max(), rho_amb.max())*10

fig, ax1 = plt.subplots()

plt.setp(ax1.get_xticklabels(), fontsize = FONTSIZE-2)
plt.setp(ax1.get_yticklabels(), fontsize = FONTSIZE-2)
ax1.set_xlabel(r'Elapsed Time [Myrs]', fontsize=FONTSIZE)
ax1.set_ylabel(r'log$_{10}(n_{gas})$ [cm$^{-3}$]', fontsize=FONTSIZE)
ax1.plot(t_ideal, nc_ideal, '.-', label='ideal MHD')
ax1.plot(t_amb, nc_amb, '.-', label='non-ideal MHD')
ax1.set_yscale('log')
ax1.tick_params(axis='y')
ax1.legend(fontsize =FONTSIZE -4)
ax1.grid(True, which='both', alpha=ALPHA)

ax2 = ax1.twinx()
plt.setp(ax2.get_xticklabels(), fontsize = FONTSIZE-2)
plt.setp(ax2.get_yticklabels(), fontsize = FONTSIZE-2)

ax2.set_ylabel(r'log$_{10}(\rho_{gas})$ [g cm$^{-3}$]', fontsize=FONTSIZE)
ax2.set_ylim(_min, _max)
ax2.set_yscale('log')

fig.tight_layout()
plt.savefig('./series/core_den.png', dpi=300)
plt.show()
plt.close(fig)


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

    return clst, dlst, tlst, slst, file_hdf5, subdirectory

def declare_globals_and_constants():

    # global variables that can be modified from anywhere in the code
    global __rloc__, __sample_size__, __input_case__, __start_snap__
    global __dense_cloud__,__threshold__, __alloc_slots__
    global FloatType, FloatType2, IntType

    # amb:   t > 3.0 Myrs snap > 225    
    # ideal: t > 3.0 Myrs snap > 270
    __start_time__  = 0.0 # Myrs
    if __input_case__ =='ideal':
        __start_snap__  = '000'

    if __input_case__ =='amb':
        __start_snap__  = '000'

    # rename
    FloatType          = np.float64
    FloatType2         = np.float128
    IntType            = np.int32

    os.makedirs("series", exist_ok=True)

    return None

input_file = sys.argv[1]
print("IC file is: ", input_file)
print("within inputs/ dir: ", input_file.split('/')[-1])

FLAG0 = "--lines" 
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
    clst, dlst, tlst, slst, file_hdf5, subdirectory = match_files_to_data(__input_case__)
    _id_ = str(input_file.split('.')[0][0] + input_file.split('.')[0][-1])
    print("ID of series.py run is", _id_)
    
    assert len(file_hdf5) == len(clst) == len(tlst), "Arrays must all have the same length"

    for each, filename in enumerate(file_hdf5):
        center   = clst[each, :]
        _time    = tlst[each]

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
            ["__rloc__", f"{__rloc__}" + " pc"],
            ["__dense_cloud__", f"{__dense_cloud__}"+ " cm^-3"],
            ]

        # I want to answer the question, why does x_input contain so few points with densities below the density peak?

        CoreDensity = dlst[each]
        X,Y,Z  = Pos[:,0], Pos[:,1], Pos[:,2]
        mask_sphere =  X*X + Y*Y+ Z*Z < __rloc__*__rloc__
        
        mask_thresh2 = Density > 1e+2
        mask_thresh4 = Density > 1e+4
        mask_thresh6 = Density > 1e+6
        mask_thresh8 = Density > 1e+8
        mask_thresh10 = Density > 1e+10
        mask_thresh12 = Density > 1e+12
        mask_thresh14 = Density > 1e+14

        mask2 = np.logical_and(mask_sphere, mask_thresh2)
        mask4 = np.logical_and(mask_sphere, mask_thresh4)
        mask6 = np.logical_and(mask_sphere, mask_thresh6)
        mask8 = np.logical_and(mask_sphere, mask_thresh8)
        mask10 = np.logical_and(mask_sphere, mask_thresh10)
        mask12 = np.logical_and(mask_sphere, mask_thresh12)
        mask14 = np.logical_and(mask_sphere, mask_thresh14)

        total_cells = Density[mask_sphere].shape[0]
        print(total_cells)

        print("\nSnapthot: ", snap)
        print("Total Cells: ", total_cells, "\n")
        print("Fraction of cells above $10^2$: ", Density[mask2].shape[0]/total_cells)
        print("Fraction of cells above $10^4$: ", Density[mask4].shape[0]/total_cells)
        print("Fraction of cells above $10^6$: ", Density[mask6].shape[0]/total_cells)
        print("Fraction of cells above $10^8$: ", Density[mask8].shape[0]/total_cells)
        print("Fraction of cells above $10^{10}$: ", Density[mask10].shape[0]/total_cells)
        print("Fraction of cells above $10^{12}$: ", Density[mask12].shape[0]/total_cells)
        print("Fraction of cells above $10^{14}$: ", Density[mask14].shape[0]/total_cells)

        total_volume = np.sum(Volume[mask_sphere])
        print("Fraction of Volume above $10^2$: ", np.sum(Volume[mask2])/total_volume)
        print("Fraction of Volume above $10^4$: ", np.sum(Volume[mask4])/total_volume)
        print("Fraction of Volume above $10^6$: ", np.sum(Volume[mask6])/total_volume)
        print("Fraction of Volume above $10^8$: ", np.sum(Volume[mask8])/total_volume)
        print("Fraction of Volume above $10^{10}$: ", np.sum(Volume[mask10])/total_volume)
        print("Fraction of Volume above $10^{12}$: ", np.sum(Volume[mask12])/total_volume)
        print("Fraction of Volume above $10^{14}$: ", np.sum(Volume[mask14])/total_volume, "\n\n")

        import matplotlib.pyplot as plt
        import numpy as np

        thresholds = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14]
        labels = [r"$10^{2}$", r"$10^{4}$ ", r"$10^{6}$ ", r"$10^{8}$ ", r"$10^{10}$ ", r"$10^{12}$ ", r"$10^{14}$ "]
        masks = [mask2, mask4, mask6, mask8, mask10, mask12, mask14]

        total_cells = Density[mask_sphere].shape[0]
        total_volume = np.sum(Volume[mask_sphere])

        cell_fractions   = [Density[m].shape[0] / total_cells      for m in masks]
        volume_fractions = [np.sum(Volume[m])   / total_volume     for m in masks]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Case {subdirectory}\n Snapshot: {snap}", fontsize=13)

        for ax, fractions, title, color in zip(
            axes,
            [cell_fractions, volume_fractions],
            ["Fraction of Cells above Density Threshold",
            "Fraction of Volume above Density Threshold"],
            ["steelblue", "darkorange"],
        ):
            ax.plot(range(len(thresholds)), fractions, marker="o", color=color, linewidth=2)
            ax.set_xticks(range(len(thresholds)))
            ax.set_xticklabels(labels)
            ax.set_xlabel("Density Threshold")
            ax.set_ylabel("Fraction")
            ax.set_title(title)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"./series/density_fractions{_id_[0]}{snap}.png", dpi=150)
        plt.show()