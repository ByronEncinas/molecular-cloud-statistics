import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data: normally distributed x, y, z values
N = 1000
x = np.random.normal(0, 1, N)
y = np.random.normal(0, 1, N)
z = np.random.normal(0, 1, N)

# Calculate the distance from the origin for each point
distance = np.sqrt(x**2 + y**2 + z**2)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colormap based on the distance from origin
sc = ax.scatter(x, y, z, c=distance, cmap='plasma')

# Add colorbar to show the scale
plt.colorbar(sc, label='Distance from Origin')

# Show plot
plt.show()
