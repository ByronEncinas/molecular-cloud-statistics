from library import *
import numpy as np

radius_vector = np.array(np.load("arepo_output_data/ArePositions.npy", mmap_mode='r'))
distance = np.array(np.load("arepo_output_data/ArepoTrajectory.npy", mmap_mode='r'))
bfield   = np.array(np.load("arepo_output_data/ArepoMagneticFields.npy", mmap_mode='r'))

#index_peaks, global_info = pocket_finder(bfield) # this plots
pocket, global_info = pocket_finder(bfield, 0.0, plot=False) # this plots
index_pocket, field_pocket = pocket[0], pocket[1]


import numpy as np


# Example usage
if __name__ == "__main__":
    
    index_shape = radius_vector.shape[:-1]
    print(index_shape)

    random_index = tuple(np.random.randint(dim) for dim in index_shape)
    
    # Select and return the vector at the random index
    x_init = radius_vector[random_index]
    
    # Find x_init in radius_vector
    indices = find_vector_in_array(radius_vector, x_init)

    lmn     = find_vector_in_array(radius_vector, x_init)
    
    print("Indices of x_init in radius_vector:", *indices[0], lmn)

    print(find_insertion_point(distance, distance[random_index]), distance[find_insertion_point(distance, distance[random_index])])
    print(distance[max(0, random_index[0]-1): min(len(distance), random_index[0] + 1)])