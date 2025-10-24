import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True

# Example scatter data
x = np.random.normal(0, 1, 10000)
y = np.random.normal(0, 1, 10000)
z = np.sqrt(x**2 + y**2)  # Some scalar field

# Create 2D binned statistic (like a projection or heatmap)
stat, x_edges, y_edges, _ = binned_statistic_2d(
    x, y, z, statistic='mean', bins=100)

# Plot projection (heatmap)
plt.figure(figsize=(6, 6))
plt.imshow(stat.T, origin='lower',
           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
           aspect='equal', cmap='plasma')
plt.colorbar(label=r"$\langle z \rangle$")

# Optionally overlay scatter points
plt.scatter(x, y, s=1, c='white', alpha=0.05)

plt.xlabel(r"$\eta$", fontsize= 16)
plt.ylabel(r"$\xi$", fontsize= 16)
plt.title(r"$Proj(x,y)$", fontsize= 16)
plt.savefig(r"./proj.png", dpi=300)


# Add table explanation
table_data = [
    ["Color", "Model"],
    ["Saturated", "H"],
    ["Unsaturated", "L"],
]

table = plt.table(cellText=table_data,
                  loc='bottom',
                  bbox=[0.2, -0.45, 0.6, 0.25])  # [x, y, width, height]
table.auto_set_font_size(False)
table.set_fontsize(9)

plt.subplots_adjust(bottom=0.3)  # make room for table
plt.savefig('./proj.png', dpi=300)