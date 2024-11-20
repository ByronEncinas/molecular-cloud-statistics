import os
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

from library import *

start_time = time.time()

max_cycles = int(sys.argv[1])

# Create a figure and axes
fig, axs = plt.subplots(1, 1, figsize=(8, 6))

# Generate a colormap with distinct colors for each cycle
cmap = cm.get_cmap('viridis', max_cycles)  # Use 'viridis' or any colormap you prefer

for cycle in range(max_cycles):
    # Load the data for this cycle
    radius_vector = np.array(np.load(f"arepo_npys/ArePositions{cycle}.npy", mmap_mode='r'))
    distance = np.array(np.load(f"arepo_npys/ArepoTrajectory{cycle}.npy", mmap_mode='r'))
    bfield = np.array(np.load(f"arepo_npys/ArepoMagneticFields{cycle}.npy", mmap_mode='r'))
    numb_density = np.array(np.load(f"arepo_npys/ArepoNumberDensities{cycle}.npy", mmap_mode='r'))

    # Plot each cycle with a unique color from the colormap and label
    axs.plot(distance, bfield/10e-6, linestyle="--", color=cmap(cycle), label=f"Cycle {cycle+1}")

# Add legend
axs.legend(title="Simulation Cycle", loc='best')  # Adjust title and location as needed
axs.set_xlabel("s (cm)")
axs.set_ylabel("$B(s)$ $\mu$G (cgs)")
axs.set_title("Magnetic Field")
axs.grid(True)

# Save the figure
plt.savefig(f"shapes_multiple{max_cycles}.png")

# Close the plot
plt.close(fig)


		
if True:
        
    from matplotlib import cm
    from matplotlib.colors import Normalize
    # Assuming mag_field is of shape (N+1, m)
    norm = Normalize(vmin=np.min(bfield), vmax=np.max(bfield))
    cmap = cm.viridis  # Choose a colormap

    ax = plt.figure().add_subplot(projection='3d')
    radius_vector /= 3.086e+18

    for k in range(max_cycles):

        # Load the data for this cycle
        radius_vector = np.array(np.load(f"arepo_npys/ArePositions{k}.npy", mmap_mode='r'))
        distance = np.array(np.load(f"arepo_npys/ArepoTrajectory{k}.npy", mmap_mode='r'))
        bfield = np.array(np.load(f"arepo_npys/ArepoMagneticFields{k}.npy", mmap_mode='r'))
        numb_density = np.array(np.load(f"arepo_npys/ArepoNumberDensities{k}.npy", mmap_mode='r'))

        x=radius_vector[:, 0]
        y=radius_vector[:, 1]
        z=radius_vector[:, 2]
        
        for l in range(len(x)):
            color = cmap(norm(bfield[l]))
            ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color,linewidth=0.3)

        #ax.scatter(x_init[0], x_init[1], x_init[2], marker="v",color="m",s=10)
        ax.scatter(x[0], y[0], z[0], marker="x",color="black",s=3)
        ax.scatter(x[-1], y[-1], z[-1], marker="x", color="black",s=3)
            
    radius_to_origin = np.sqrt(x**2 + y**2 + z**2)
    zoom = np.max(radius_to_origin)/2
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
    cbar.set_label('Magnetic Field Strength')

    plt.savefig(f'field_shapes/MagneticFieldTopology.png', bbox_inches='tight')
    plt.show()
