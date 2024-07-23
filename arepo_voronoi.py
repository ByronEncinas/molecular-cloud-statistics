from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np

rng = np.random.default_rng()
pts = rng.random((10, 3))   # 30 random pts in 2-D
print(type(pts))

print(pts)

hull = ConvexHull(pts, qhull_options="QJ")

import matplotlib.pyplot as plt

plt.plot(pts[:,0], pts[:,1], 'o')
for simplex in hull.simplices:
    print(simplex)
    plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-')
plt.plot(pts[hull.vertices,0], pts[hull.vertices,1], 'r--', lw=2)
plt.plot(pts[hull.vertices[0],0], pts[hull.vertices[0],1], 'ro')

# if dim < 2 then: perimeter, else: area
print(hull.area) 
# if dim < 2 then: area, else: volume
print(hull.volume) 

plt.show()

hull = ConvexHull(pts, qhull_options="QJ")

plt.plot(pts[:,0], pts[:,1], 'o')
for simplex in hull.simplices:
    print(simplex)
    plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-')
plt.plot(pts[hull.vertices,0], pts[hull.vertices,1], 'r--', lw=2)
plt.plot(pts[hull.vertices[0],0], pts[hull.vertices[0],1], 'ro')

# if dim < 2 then: perimeter, else: area
print(hull.area) 
# if dim < 2 then: area, else: volume
print(hull.volume) 

plt.show()

