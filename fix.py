from library import *
import numpy as np
import scipy

radius_vector = np.array(np.load("cluster_outputs/npysJAKAR/ArePositions.npy", mmap_mode='r'))
distance = np.array(np.load("cluster_outputs/npysJAKAR/ArepoTrajectory.npy", mmap_mode='r'))
bfield   = np.array(np.load("cluster_outputs/npysJAKAR/ArepoMagneticFields.npy", mmap_mode='r'))
numb_density   = np.array(np.load("cluster_outputs/npysJAKAR/ArepoNumberDensities.npy", mmap_mode='r'))


index_global_max = np.where(bfield == max(bfield))[0]
print(index_global_max[1:-2])
print(index_global_max)

#bfield[index_global_max[1:-2]] = bfield[index_global_max[1:-2]]/2

#index_peaks, global_info = pocket_finder(bfield) # this plots
pocket, global_info = pocket_finder(bfield, 0.0, plot=True) # this plots
index_pocket, field_pocket = pocket[0], pocket[1]
print(index_pocket)
print(field_pocket)

import numpy as np
import random

# Example usage
if __name__ == "__main__":
    
    index_shape = bfield.shape[:-1]
    
    random_index = len(bfield)//2 + 1
    
    print(random_index)
    # Select and return the vector at the random index
    ijk = np.argmax(bfield)
    print("argmax",ijk)
    B_init = bfield[random_index]
    
    print(index_shape, B_init)
    # Find x_init in radius_vector
    
    lmn     = np.where(bfield == B_init)
    
    print("Indices of x_init in radius_vector:", lmn)
    print("B at x_init:", bfield[lmn], "inserted between:", bfield[max(0, random_index): min(len(distance), random_index + 2)])

    print(find_insertion_point(distance, distance[random_index]), distance[find_insertion_point(distance, distance[random_index])])
    print(distance[max(0, random_index-1): min(len(distance), random_index + 1)])

    if True:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        axs[0].plot(distance, bfield, linestyle="--", color="m")
        axs[0].scatter(distance[ijk], bfield[ijk], marker="x", color="g")
        axs[0].scatter(distance, bfield, marker="+", color="m")
        axs[0].set_xlabel("trajectory (pc)")
        axs[0].set_ylabel("$B(s)$ (cgs units )")
        axs[0].set_title("Individual Magnetic Field Shape")
        #axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(distance, numb_density, linestyle="--", color="m")
        axs[1].set_xlabel("trajectory (cgs units Au)")
        axs[1].set_ylabel("$n_g(s)$ Field (cgs units $M_{sun}/pc^3$) ")
        axs[1].set_title("Gas Density along Magnetic Lines")

        #axs[1].legend()
        axs[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig("_field-density_shape.png")

        # Show the plot
        #plt.show()
        plt.close(fig)