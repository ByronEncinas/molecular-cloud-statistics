from library import *
import numpy as np

radius_vector = np.array(np.load("arepo_output_data/ArePositions.npy", mmap_mode='r'))
distance = np.array(np.load("arepo_output_data/ArepoTrajectory.npy", mmap_mode='r'))
bfield   = np.array(np.load("arepo_output_data/ArepoMagneticFields.npy", mmap_mode='r'))

#index_peaks, global_info = pocket_finder(bfield) # this plots
pocket, global_info = pocket_finder(bfield, 0.0, plot=False) # this plots
index_pocket, field_pocket = pocket[0], pocket[1]


import numpy as np
import random

# Example usage
if __name__ == "__main__":
    
    index_shape = bfield.shape[:-1]
    
    random_index = len(bfield)//2
    
    print(random_index)
    # Select and return the vector at the random index
    B_init = bfield[random_index]
    
    print(index_shape, B_init)
    # Find x_init in radius_vector
    
    lmn     = np.where(bfield == B_init)
    
    print("Indices of x_init in radius_vector:", lmn)
    print("B at x_init:", bfield[lmn], "inserted between:", bfield[max(0, random_index-1): min(len(distance), random_index + 1)])

    print(find_insertion_point(distance, distance[random_index]), distance[find_insertion_point(distance, distance[random_index])])
    print(distance[max(0, random_index-1): min(len(distance), random_index + 1)])