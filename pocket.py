import os, sys, time
import src.tmp_library as tmplib
from mpi4py import MPI          # Move to TOP of __main__
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import warnings

def translation_rotation(x, b, v, p=False, ax=None):
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

if True:
    n = 3

    # generate mesh
    X, Y, Z = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-2, 2, n), np.linspace(-2, 2, 2*n))

    # mask to get a cilinder, not needed xD
    mk = X**2 + Y**2 < 3.0

    # apply
    X, Y, Z = X[mk], Y[mk], Z[mk]

    # stack into a vector (N, 3)
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # traslation to x
    _x = np.array([1.0, 1.0, 1.0]) * 3

    # orientation of z' acis in canonical basis
    _b = np.array([1.0, 1.0, 1.0])  

    # points after both operationstranslation_rotation
    pointsp = translation_rotation(_x, _b, points, p=True)


start_time = time.time()

input_file = sys.argv[1]

tmplib.config_ic(input_file)

print(f"\n[{sys.argv[0]}]: started running...\n", flush=True)
print(f"[__input_case__] ", tmplib.__input_case__,flush=True)
# amb:   t > 3.0 Myrs snap > 225    
# ideal: t > 3.0 Myrs snap > 270
__start_time__  = 3.0 # Myrs

if tmplib.__input_case__ =='ideal':
    __start_snap__  = '270'

if tmplib.__input_case__ =='amb':
    __start_snap__  = '225'

if tmplib.FLAG2 in sys.argv:
    __start_snap__  = '000'

os.makedirs("series", exist_ok=True)

if __name__=='__main__':

    clst, dlst, tlst, slst, file_hdf5 = tmplib.match_files_to_data(tmplib.__input_case__,__start_snap__)
    _id_ = str(input_file.split('.')[0][0] + input_file.split('.')[0][-1])
    if tmplib.FLAG1 in sys.argv:
        _id_ = str(input_file.split('.')[0][0] + input_file.split('.')[0][-3:])
    print("ID of series.py run is", _id_)

    
    assert len(file_hdf5) == len(clst) == len(tlst), "Arrays must all have the same length"

    survivors_fraction = np.zeros(file_hdf5.shape[0])
    
    df_stats = dict()
    df_fields= dict()
    
    __sample_size__ = 5


    for each in range(len(file_hdf5)):
        filename = file_hdf5[each]
        center   = clst[each, :]
        _time    = tlst[each]
        
        tmplib.config_arepo(filename, center)

        table_data = [
            ["__curr_snap__", filename.split('/')[-1]],
            ["__cores_avail__", os.cpu_count()],
            ["__rloc__", f"{tmplib.__rloc__}" + " pc"],
            ["__threshold__", f"{tmplib.__threshold__}"+ " cm^-3"],
            ["__dense_cloud__", f"{tmplib.__dense_cloud__}"+ " cm^-3"],
            ["__alloc_slots__", tmplib.__alloc_slots__],
            ["__sample_size__", tmplib.__sample_size__]
            ]

        print(tabulate(table_data, headers=["Property", "Value"], tablefmt="grid"), '\n', flush=True)

        tree = cKDTree(tmplib.Pos)

        try:
            x_input    = tmplib.uniform_in_3d_tree_dependent(tree, tmplib.__sample_size__, rloc=tmplib.__rloc__, n_crit=tmplib.__dense_cloud__)   
        except Exception as e:
            warnings.warn(f"[snap={tmplib.snap}]", e)
            warnings.warn(f"[snap={tmplib.snap}] At current snapshots, no cloud above {tmplib.__dense_cloud__} cm-3", Warning)
            print(f"[Snap] snap {filename}: skipping", flush=True)
            tmplib.config_arepo(filename, center, True)
            continue

        try:
            radius_vectors, magnetic_fields, numb_densities, follow_index, path_column, survivors1 = tmplib.crs_path(x_init=x_input, n_crit=tmplib.__threshold__)
            assert np.any(numb_densities > tmplib.__threshold__), f"No values above threshold {tmplib.__threshold__} cm-3"
        except Exception as e:
            warnings.warn(f"[snap={tmplib.snap}]", e)
            print(f"[CRS] Invalid result from intergration: {tmplib.snap}: skipping", flush=True)
            tmplib.config_arepo(filename, center, True)
            continue

        print("__alloc_slots__: ", tmplib.__alloc_slots__, flush=True)
        r_u, n_rs, B_rs, survivors2 = tmplib.eval_reduction(magnetic_fields, numb_densities, follow_index, tmplib.__threshold__)
        
        pocket_question_mark = r_u[r_u<1].shape[0] != 0
        
        # if question_mark is different to zero, then we will generate a cilinder in the field direction.
        


