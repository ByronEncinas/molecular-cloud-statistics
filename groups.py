import matplotlib
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from matplotlib.ticker import MaxNLocator
import csv, glob, os, sys, time, h5py
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib as mpl
import pandas as pd
import numpy as np
import warnings
import pickle

#from src.library import *

FloatType          = np.float64
FloatType2         = np.float128
IntType            = np.int32

mpl.rcParams['text.usetex'] = True
MARKERS = ['v', 'o']
COLORS = [
    "#8E2BAF",  # Deep Purple     — original
    "#148A02",  # Forest Green    — original
    "#C42B8E",  # Magenta Rose    — analogous to purple, warm bridge
    "#AF2B2B",  # Crimson         — bold warm anchor
    "#D4820A",  # Amber           — energetic warm accent
    "#8AAF0A",  # Chartreuse      — yellow-green, bridges to green
    "#0A8A5E",  # Emerald Teal    — cool-green bridge
    "#0A5EAF",  # Royal Blue      — cool counterweight
]
ALPHA   = 0.9
SIZE    = 8
FONTSIZE = 18
GRID_ALPHA = 0.5
CASE_COLOR  = {"ideal": "steelblue", "amb": "darkorange"}
CASE_LABEL  = {"ideal": "ideal MHD", "amb": "+ ambipolar diffusion"}
MARKER_INIT = "o"
MARKER_LAST = "X"
MARKER_SIZE = 60

start_time = time.time()

cases = ['ideal', 'amb']
COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]
#, float_precision="round_trip"
data = {case: pd.read_csv(f'./util/{case}_clouds.csv') for case in cases}

def match_data_to_files(__input_case__, range_function = np.array(list(range(0,500)))):
    """
    range_function : contains a list with all available positions in of each cloud
    """
    if __input_case__ == 'ideal':
        subdirectory = 'ideal_mhd'
        file_hdf5 = sorted(np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5')))
    elif __input_case__ == 'amb':
        subdirectory = 'ambipolar_diffusion'
        file_hdf5 = sorted(np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5')))

    #assert os.path.exists(file_hdf5), f"[{file_hdf5} cloud data not present]"

    available_snapshot_files = np.zeros_like(range_function) == 1

    for snap_file in file_hdf5:
        snap_val = snap_file.split(".")
        available_snapshot_files[int(snap_val[-2][-3:])] = True
        
    return available_snapshot_files, file_hdf5

def cloud_positions(df, which, cols=("c_coord_X", "c_coord_Y", "c_coord_Z")):
    agg = "last"# if which == "last" else "last"
    return (df.groupby("cloud")[list(cols)]
              .agg(agg)
              .rename_axis("cloud")
              .reset_index())

ref_case   = cases[0]   # "ideal"
other_case = cases[1]   # "amb"

# ── manually specified pairs (ref_cloud, other_cloud) ────────────────────────
manual_pairs = [
    ("cloud-0", "cloud-0"),
    ("cloud-1", "cloud-2"),
    ("cloud-2", "cloud-4"),
    ("cloud-3", "cloud-1"),
    ("cloud-4", "cloud-5"),
    ("cloud-5", "cloud-3"),
]

pairs = []
for cn_ref, cn_other in manual_pairs:
    pts_ref   = data[ref_case  ][data[ref_case  ]["cloud"] == cn_ref  ][COORD_COLS].values
    pts_other = data[other_case][data[other_case]["cloud"] == cn_other][COORD_COLS].values
    n = min(len(pts_ref), len(pts_other))
    dist = np.linalg.norm(pts_ref[:n] - pts_other[:n], axis=1).mean()
    pairs.append((cn_ref, cn_other, dist))

dists = np.array([p[2] for p in pairs])
threshold = dists.mean() + dists.std()

pairs_filtered = [p for p in pairs if p[2] <= threshold]
pairs_removed  = [p for p in pairs if p[2] >  threshold]

pairs = pairs_filtered

data_cloud_pairs = dict()

for i, (ref_name, other_name, dist) in enumerate(pairs):
    ideal_data = data[ref_case]
    amb_data = data[other_case]
    #print(dist)
    data_cloud_pairs[i] = (ideal_data[ideal_data['cloud'] == ref_name], amb_data[amb_data['cloud'] == other_name])

with open('./util/cloud_matches.pkl', 'wb') as f:
    pickle.dump(data_cloud_pairs, f)

"""
data_cloud_pairs

First  dimesnion: 0-3 The cloud pairs
Second dimesnion: 0-1 The simulations (ideal, non-ideal)
Third dimesnion: [snap, cloud, time_value, c_coord_X, c_coord_Y, c_coord_Z, Peak_Density] columns of dataframe NOTE: 'cloud' column nust be constant

print(data_cloud_pairs[0][0][COORD_COLS])

"""

#with open('./util/cloud_matches.pkl', 'rb') as f:
#    data_cloud_pairs = pickle.load(f)

ideal_mask, ideal_hdf5 = match_data_to_files(ref_case, range_function = np.array(list(range(0,500))))
amb_mask, amb_hdf5   = match_data_to_files(other_case, range_function = np.array(list(range(0,500))))

#print(np.where(ideal_mask == True))
#print(np.where(amb_mask == True))

CLOUD_SNAP_CENTER = ["snap"] + COORD_COLS

# iterate over all 4 clouds, on both simulations
for index_cloud, cloud_data in data_cloud_pairs.items():
    if index_cloud == 0:
        continue

    # DataFrame for ideal cloud data
    ideal_data = cloud_data[0].sort_values("snap")[CLOUD_SNAP_CENTER]

    # iterate over all snaps
    for snap, x, y, z in ideal_data[ideal_mask].to_numpy():
        print("ID: ", snap, x, y, z, index_cloud)
        #compute_reduction_factor_in_parallel()

    # DataFrame for non-ideal cloud data
    amb_data = cloud_data[1].sort_values("snap")[CLOUD_SNAP_CENTER]

    # iterate over all snaps
    for snap, x, y, z in amb_data[amb_mask].to_numpy():
        print("AD", snap, x, y, z, index_cloud)
        #compute_reduction_factor_in_parallel()


    



exit()
# iterate over idea and non-ideal datasets
for case, simulation_data in enumerate([ideal_data, amb_data]):
    print(simulation_data)
    break
    # iterate over snapshots with all 4 clouds data
    for coordinates in simulation_data:
        print(case, coordinates)

    break
    

if False:
    del data, data_cloud_pairs

    #with open('./util/cloud_matches.pkl', 'rb') as f:
    #    data_cloud_pairs = pickle.load(f)

if False:
    fig, ax = plt.subplot_mosaic(
        [[0,1,2,3]],
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1], "wspace": 0.0}
    )

    ax[0].set_ylabel(r"$\log_{10}(n_g / \rm{cm}^{-3})$",fontsize=FONTSIZE-4)

    min_density = np.inf
    max_density = 0.0

    for ith_cloud in data_cloud_pairs:
        ax[ith_cloud].set_xlabel(r"$t \  \rm{[Myrs]}$",fontsize=FONTSIZE-4)

        max_density = max(max(np.max(data_cloud_pairs[ith_cloud][0]["Peak_Density"]), np.max(data_cloud_pairs[ith_cloud][1]["Peak_Density"])), max_density)
        min_density = min(min(np.min(data_cloud_pairs[ith_cloud][0]["Peak_Density"]), np.min(data_cloud_pairs[ith_cloud][1]["Peak_Density"])), min_density)
        #print("IDEAL:\n",data_cloud_pairs[ith_cloud][0]["Peak_Density"], "\nNON-IDEAL:\n",data_cloud_pairs[ith_cloud][1]["Peak_Density"])
        ax[ith_cloud].plot(data_cloud_pairs[ith_cloud][0]["time_value"], np.log10(data_cloud_pairs[ith_cloud][0]["Peak_Density"]), "-",label = "ideal MHD", color="black") #OLORS[ith_cloud])
        ax[ith_cloud].plot(data_cloud_pairs[ith_cloud][1]["time_value"], np.log10(data_cloud_pairs[ith_cloud][1]["Peak_Density"]), "--",label = "non-ideal MHD", color="black") #COLORS[ith_cloud])

    print(min_density, max_density)

    min_density = np.log10(min_density) # -0.1
    max_density = np.log10(max_density) # 4 #

    ax[3].legend(fontsize=FONTSIZE-4)

    #ax[0].set_xlim(-0.01, 0.1)
    #ax[1].set_xlim(-0.01, 0.1)
    #ax[2].set_xlim(-0.01, 0.1)
    #ax[3].set_xlim(-0.01, 0.1)


    ax[0].set_ylim(min_density, max_density)
    ax[1].set_ylim(min_density, max_density)
    ax[2].set_ylim(min_density, max_density)
    ax[3].set_ylim(min_density, max_density)

    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[2].xaxis.set_major_locator(MaxNLocator(nbins=3)) 
    ax[3].xaxis.set_major_locator(MaxNLocator(nbins=3)) 

    plt.savefig(f'./others/nvt.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# ideal_mask and amb_mask filter snapshots to data

#print(data_cloud_pairs) # <- start again trying to match this to files

cloud_coords = {}  # key: ith_cloud, value: (2, 500, 3) array

for ith_cloud in data_cloud_pairs:
    ideal_df = data_cloud_pairs[ith_cloud][0].sort_values("snap") # sort_values("snap", ascending=False)
    amb_df   = data_cloud_pairs[ith_cloud][1].sort_values("snap")

    ideal_coords = ideal_df[COORD_COLS].to_numpy()  # (500, 3)
    amb_coords   = amb_df[COORD_COLS].to_numpy()    # (500, 3)

    cloud_coords[ith_cloud] = np.stack([ideal_coords, amb_coords], axis=0)  # (2, 500, 3)

print(ideal_hdf5)
print(cloud_coords[0][0, ideal_mask,:])
print(amb_hdf5)
print(cloud_coords[1][0, amb_mask,:])


print(np.sum(ideal_mask), np.sum(amb_mask))



# iterate over this each cloud that was matched between both simulations
# save only cloud coordinates coinciding with available files
ith_ideal_cloud_coords = dict()
ith_amb_cloud_coords   = dict()

print(ideal_cloud_times)
print(amb_cloud_times)

print(data_cloud_pairs[0][0][COORD_COLS].to_numpy()[ideal_mask,:] )
"""
for snapshot in ideal_hdf5:
    print(snapshot)

for snapshot in amb_hdf5:
    print(snapshot)
"""

exit()

for ith_cloud in data_cloud_pairs:
    # ideal data - positions, and time
    ith_ideal_cloud_coords[ith_cloud] = data_cloud_pairs[ith_cloud][0][COORD_COLS].to_numpy()#[ideal_mask,:]  # shape: (N, 3)

    # non-ideal data - positions, and time
    ith_amb_cloud_coords[ith_cloud]   = data_cloud_pairs[ith_cloud][1][COORD_COLS].to_numpy()#[amb_mask,:]


# iterate over available hdf5 files, to trace magnetic field lines
for snapshot, (ideal_clouds_hdf5, time) in enumerate(zip(ideal_hdf5, ideal_cloud_times)):
    print(snapshot, ideal_clouds_hdf5)
    for ith_cloud, cloud_center in ith_ideal_cloud_coords.items():
        print(ith_cloud, cloud_center[ith_cloud])
        

for snapshot, (amb_clouds_hdf5, time) in enumerate(zip(amb_hdf5, amb_cloud_times)):
    print(snapshot, amb_clouds_hdf5)
    for ith_cloud, cloud_center in ith_amb_cloud_coords.items():
        print(ith_cloud, cloud_center)



from multiprocessing import Pool, cpu_count

