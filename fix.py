from library import *
import numpy as np

distance = np.array(np.load("arepo_output_data/ArepoTrajectory.npy", mmap_mode='r'))
bfield   = np.array(np.load("arepo_output_data/ArepoMagneticFields.npy", mmap_mode='r'))

#index_peaks, global_info = pocket_finder(bfield) # this plots
pocket, global_info = pocket_finder(bfield, 0.0, plot=True) # this plots
index_pocket, field_pocket = pocket[0], pocket[1]
