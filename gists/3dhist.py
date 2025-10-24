from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data_2d = [ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [11, 12, 13, 14, 15, 16, 17, 18 , 19, 20],
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30] ]


#
# Convert it into an numpy array.
#
data_array = np.array(data_2d)
print(data_array.shape)
#
# Create a figure for plotting the data as a 3D histogram.
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
# Create an X-Y mesh of the same dimension as the 2D data. You can
# think of this as the floor of the plot.
#
x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),
                              np.arange(data_array.shape[0]) )

print(x_data.shape, y_data.shape)
#
# Flatten out the arrays so that they may be passed to "ax.bar3d".
# Basically, ax.bar3d expects three one-dimensional arrays:
# x_data, y_data, z_data. The following call boils down to picking
# one entry from each array and plotting a bar to from
# (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
#
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = data_array.flatten()
ax.bar3d( x_data,
          y_data,
          np.zeros(len(z_data)),
          1, 1, z_data )
#
# Finally, display the plot.
#
plt.savefig('./hist3d.png')
plt.close()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data: 4 different normal distributions
np.random.seed(0)
data = [
    np.random.poisson(lam=2, size=1500),
    np.random.poisson(lam=2, size=1500),
    np.random.poisson(lam=6, size=1500),
    np.random.poisson(lam=6, size=1500),
    np.random.poisson(lam=6, size=1500),
    np.random.poisson(lam=6, size=1500),
    np.random.poisson(lam=6, size=1500)

]

def hist3d(data):
    colors = ['yellow', 'blue', 'green', 'red','yellow', 'blue', 'green', 'red']

    # Set up 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each histogram at a different Y position
    bins_ = int(np.ceil(np.sqrt(np.mean([a.shape[0] for a in data]))))
    for i, (d, color) in enumerate(zip(data, colors)):
        counts, bins = np.histogram(d, bins=bins_)
        xs = 0.25 * (bins[:-1] + bins[1:])
        print(bins)
        ys = np.full_like(xs, i * 5)  # Stack along Y
        zs = np.zeros_like(xs)

        dx = (bins[1] - bins[0]) * 0.8  # Thinner bars in X
        dy = 0.1                 # Thinner depth in Y
        dz = counts

        ax.bar3d(xs, ys, zs, dx, dy, dz, color=color, alpha=0.6)

    # Labels and aesthetics
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Stacked 3D Histograms')
    ax.view_init(elev=20, azim=-60)  # adjust angle to match reference image
    plt.tight_layout()
    plt.savefig('./hist3d_proj.png')
    plt.close()

hist3d(data)