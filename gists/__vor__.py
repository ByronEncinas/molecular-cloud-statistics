import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np

pc_to_au = (3.086 * 1.0e+18)/(1.496 * 1.0e+13)

out_radius = 1.0e-3
frame_radius = out_radius*pc_to_au/2
dz = 3.e-5
dz = 0.1
z_center = 0.0

def __vor__(VP):
  
    mask0 = VP[:,0]*VP[:,0] + VP[:,1]*VP[:,1]+VP[:,2]*VP[:,2] < out_radius**2
    coords = VP[mask0,:]
    mask = (coords[:, 2] > z_center - dz/2) & (coords[:, 2] < z_center + dz/2)
    points_2d = coords[mask][:, :2] *pc_to_au # x and y positions only

    vor = Voronoi(points_2d)
    voronoi_plot_2d(vor, show_points =False, show_vertices=False, line_colors='black', line_width=0.3)
    plt.xlim(-frame_radius, frame_radius)
    plt.ylim(-frame_radius, frame_radius)

    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.tight_layout()
    plt.savefig('./voronoi_2d_color.png')

def __color_vor__(VP,D,var=''):
    
    mask0 = VP[:,0]**2 + VP[:,1]**2 + VP[:,2]**2 < out_radius**2
    coords = VP[mask0,:]
    mask = (coords[:, 2] > z_center - dz/2) & (coords[:, 2] < z_center + dz/2)
    points_2d = coords[mask][:, :2]*pc_to_au  # x and y positions only

    densities = D[mask0][mask]  # assuming Density array matches VP shape
    vor = Voronoi(points_2d)
    polygons = []
    colors = []

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue  # skip infinite regions
        polygon = [vor.vertices[i] for i in region]
        polygons.append(Polygon(polygon))
        colors.append(densities[point_idx])  # color by density
    fig, ax = plt.subplots()

    if var == 'D':
        norm = LogNorm(vmin=np.min(colors), vmax=np.max(colors))
        p = PatchCollection(polygons, array=np.array(colors), cmap='viridis', edgecolor=None, norm=norm)
        ax.add_collection(p)
        cbar = plt.colorbar(p)
        cbar.set_label(r'$\log(n_g/\rm{cm}^{-3})$')
    elif var == 'B':
        norm = LogNorm(vmin=np.min(colors), vmax=np.max(colors))
        p = PatchCollection(polygons, array=np.array(colors), cmap='viridis', edgecolor=None, norm=norm)
        ax.add_collection(p)
        cbar = plt.colorbar(p)
        cbar.set_label(r'$\log(B/ \mu \rm{G})$')
    elif var == 'V':
        norm = Normalize(vmin=min(colors), vmax=max(colors))
        p = PatchCollection(polygons, array=np.array(colors), cmap='viridis', edgecolor=None, norm=norm)

        ax.add_collection(p)
        cbar = plt.colorbar(p)
        cbar.set_label(r'$\log(v/\rm{km} \rm{s}^{-1})$')
    ax.set_xlim(-frame_radius, frame_radius)
    ax.set_ylim(-frame_radius, frame_radius)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    plt.savefig(f'./{var}voronoi_2d.png', dpi=300)

def traslation_rotation(x, b, v, p=False, ax=None):
    from scipy.spatial.transform import Rotation as R
    """
    x corresponds with a value in x_input with a recorded value of r < 1
    b corresponds with the field vector at x_less
    """
    i,j,k = np.array([1.,0.,0.]),np.array([0.,1.,0.]), np.array([.0,0.,1.])
    # we use grahm schmidt to get the basis vectors perpendicular to b_less
    e1 = b / np.linalg.norm(b)               # z'
    if np.dot(i, e1) != 1:
        e2 = i - np.dot(i, e1)*e1              
        e2 /= np.linalg.norm(e2)                
    elif np.dot(j, e1)!= 1:
        e2 = j - np.dot(j, e1)*e1              
        e2 /= np.linalg.norm(e2)                
    else:
        e2 = k - np.dot(k, e1)*e1                  
        e2 /= np.linalg.norm(e2)    

    e3 = np.cross(e1, e2)                    # y'

    # now using e2 and e2 we can generate points on the 2-3 plane,
    # and follow field lines there, maybe we can be able to map r
    # in the X'Y' plane around a pockets
    Rmat = np.column_stack((e2, e3, e1))
    # rotation instance
    rot = R.from_matrix(Rmat)

    # apply rotation to the vector
    from copy import deepcopy
    t = deepcopy(x)
    t[2] = t[2] + np.max(v[:,2])
    t = t - x
    t    = rot.apply(t)
    t += x
    xpyp = rot.apply(v)
    
    if p:    
        X, Y, Z = v[:,0], v[:,1], v[:,2]
        Xp, Yp, Zp = xpyp[:,0] + x[0] , xpyp[:,1] + x[1], xpyp[:,2] + x[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X, Y, Z, marker='o', c='r', alpha=0.3, s=5)
        ax.scatter(Xp, Yp, Zp, marker='x', c='g', alpha=0.3, s=5)

        ax.scatter(x[0], x[1], x[2], marker='o', c='r', alpha=0.5, s=10)
        ax.quiver(t[0], t[1], t[2], e1[0], e1[1], e1[2], color='black', alpha=1.0, length=1)
        
        ax.view_init(elev=10, azim=-45)  # elev = vertical angle, azim = horizontal rotation
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.savefig('./XY_XpYp.png')
    
    return xpyp + x

if __name__=='__main__':
    n = 10

    # generate mesh
    X, Y, Z = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-2, 2, n), np.linspace(-2, 2, 2*n))
    
    # mask to get a cilinder, not needed xD
    #mk = X**2 + Y**2 < 3.0

    # apply
    #X, Y, Z = X[mk], Y[mk], Z[mk]

    # stack into a vector (N, 3)
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # traslation to x
    _x = np.array([1.0, 1.0, 1.0]) * 3

    # orientation of z' acis in canonical basis
    _b = np.array([1.0, 1.0, 1.0])  

    # points after both operations
    pointsp = traslation_rotation(_x, _b, points)
