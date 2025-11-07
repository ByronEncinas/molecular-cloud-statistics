"""
If you already have a list of file paths in `files_hdf5` and a function `return_stats(file)` that processes each file independently, then `multiprocessing.Pool.map` is perfect.

---

### Parallelizing `return_stats()` over `files_hdf5`

```python
from multiprocessing import Pool, cpu_count

def return_stats(file_path):
    # Your heavy analysis function
    # e.g., read HDF5, compute statistics, return dict or DataFrame
    ...
    return result


```

---

### Notes and Tips

1. **Use `if __name__ == "__main__":`**
   This is essential on Linux and *mandatory* on Windows/macOS — it prevents recursive process spawning.

2. **Control the number of processes**
   Using all CPUs (`cpu_count()`) can be too heavy if each file uses a lot of memory or internal threads (e.g., HDF5 I/O).
   You can tune it, e.g.:

   ```python
   nproc = 4  # or whatever works best
   ```

3. **If your function has multiple arguments**
   You can use `starmap` or `pool.map` with partials:

   ```python
   from functools import partial
   pool.map(partial(return_stats, param=some_value), files_hdf5)
   ```

4. **Progress bar (optional)**
   If you want to visualize progress:

   ```python
   from tqdm import tqdm
   results = list(tqdm(pool.imap(return_stats, files_hdf5), total=len(files_hdf5)))
   ```

5. **Return values**
   Whatever `return_stats()` returns will be collected in `results` as a list in the same order as `files_hdf5`.

---

### Example of a realistic use case

```python
def return_stats(file_path):
    import h5py
    import numpy as np

    with h5py.File(file_path, 'r') as f:
        data = f['density'][()]
    return {
        "file": file_path,
        "mean_density": np.mean(data),
        "std_density": np.std(data)
    }
```

Then you can easily combine all results:

```python
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

---


"""
import csv, glob, os, h5py
import numpy as np
from scipy.spatial import cKDTree
import timeseries as ts
from library import *

# global variables that can be modified from anywhere in the code
global __rloc__, __sample_size__, __input_case__, __start_snap__, __alloc_slots__, __dense_cloud__,__threshold__, N
global Pos, VoronoiPos, Bfield, Mass, Density, Bfield_grad, Density_grad, Volume

# immutable objects, use type hinting to debug if error
N               = 2_500
__rloc__        = 0.1
__sample_size__ = 1_000
__input_case__  = 'ideal'
__alloc_slots__ = 2_000
__dense_cloud__ = 1.0e+2
__threshold__   = 1.0e+2

# output directory
os.makedirs("series", exist_ok=True)

def return_stats(file):
    df_r_stats = dict()
    df_c_stats = dict()
    ts.declare_globals_and_constants()  
    num_file = file.split('.')[0][-3:]
    file_path       = f'./util/{__input_case__}_cloud_trajectory.txt'

    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
        next(csv_reader)  # Skip the header row
        print('File opened successfully')
        for row in csv_reader:
            if num_file == str(row[0]):
                center = np.array([float(row[2]),float(row[3]),float(row[4])])
                snap =str(row[0])

    snap = int(num_file)
    data = h5py.File(file, 'r')
    __Boxsize__ = data['Header'].attrs['BoxSize']
    _time = data['Header'].attrs['Time']
    Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=np.float128)
    Density = np.asarray(data['PartType0']['Density'], dtype=np.float128)*gr_cm3_to_nuclei_cm3
    VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=np.float128)
    Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=np.float128)
    Mass = np.asarray(data['PartType0']['Masses'], dtype=np.float128)
    Bfield_grad = np.zeros((len(Pos), 9))
    Density_grad = np.zeros((len(Density), 3))
    Volume   = Mass/Density
    VoronoiPos-=center
    Pos       -=center    

    for dim in range(3):
        pos_from_center = Pos[:, dim]
        too_high = pos_from_center > __Boxsize__ / 2
        too_low  = pos_from_center < -__Boxsize__ / 2
        Pos[too_high, dim] -= __Boxsize__
        Pos[too_low,  dim] += __Boxsize__

    from tabulate import tabulate
    table_data = [
        ["__snapshot__", file.split('/')[-1]],
        ["__cores_avail__", os.cpu_count()],
        ["__alloc_slots__", 2*N],
        ["__rloc__", __rloc__],
        ["__sample_size__", __sample_size__],
        ["__Boxsize__", __Boxsize__]
        ]
    
    print(tabulate(table_data, headers=["Property", "Value"], tablefmt="grid"), '\n')
    
    tree = cKDTree(Pos)

    try:
        #x_input = np.vstack([uniform_in_3d(__sample_size__, __rloc__, n_crit=__dense_cloud__), np.array([0.0,0.0,0.0])])
        x_input    = ts.uniform_in_3d_tree_dependent(tree, __sample_size__, rloc=__rloc__, n_crit=__dense_cloud__)    
    except:
        raise ValueError("[Error] Faulty generation of 'x_input'")

    directions=fibonacci_sphere(10)
    __0, __1, mean_column, median_column = ts.line_of_sight(x_init=x_input, directions=directions, n_crit=__threshold__)
    radius_vectors, magnetic_fields, numb_densities, follow_index, path_column, survivors1 = ts.crs_path(x_init=x_input, n_crit=__threshold__)

    assert np.any(numb_densities > __threshold__), f"No values above threshold {__threshold__} cm-3"

    data.close()

    r_u, n_rs, B_rs, survivors2 = ts.eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__)
    r_l, _1, _2, _3 = ts.eval_reduction(magnetic_fields, numb_densities, follow_index, __threshold__//10) # not needed

    survivors = np.logical_and(survivors1, survivors2)

    print(np.sum(survivors)/survivors.shape[0], " Survivor fraction")  

    u_input         = x_input[np.logical_not(survivors),:] 
    x_input         = x_input[survivors,:]
    radius_vectors  = radius_vectors[:, survivors, :]
    numb_densities  = numb_densities[:, survivors]
    magnetic_fields = magnetic_fields[:, survivors]
    mean_column     = mean_column[survivors]
    median_column   = median_column[survivors]
    path_column     = path_column[survivors]
    n_rs            = n_rs[survivors]
    B_rs            = B_rs[survivors] 
    r_u             = r_u[survivors]
    r_l             = r_l[survivors]

    #distance = np.linalg.norm(x_input, axis=1)*pc_to_cm

    mean_r_u, median_r_u, skew_r_u, kurt_r_u = ts.describe(r_u)
    mean_r_l, median_r_l, skew_r_l, kurt_r_l = ts.describe(r_l)

    """
    smth
    """
    c_stats_dict = {
        "time": _time,
        "x_input": x_input,
        "n_rs": n_rs,
        "B_rs": B_rs,
        "n_path": path_column,
        "ratio0": mean_column/path_column,
        "ratio1": median_column/path_column
    }
    df_c_stats[str(snap)]  = c_stats_dict

    r_stats_dict = {
        "time": _time,
        "surv_fraction": np.sum(survivors)/survivors.shape[0],
        "mean_r_u": mean_r_u,
        "median_r_u": median_r_u,
        "skew_r_u": skew_r_u,
        "kurt_r_u": kurt_r_u,
        "mean_r_l": mean_r_l,
        "median_r_l": median_r_l,
        "skew_r_l": skew_r_l,
        "kurt_r_l": kurt_r_l
    }

    df_r_stats[str(snap)]  = r_stats_dict # multiindex dataframe
    

    if 'Pos' in globals():
        print("\nPos is global")
        
    del tree

    import src.library
    src.library._cached_tree = None
    src.library._cached_pos = None

    return df_r_stats, df_c_stats

from multiprocessing import Pool, cpu_count

if __name__ == "__main__":
    if __input_case__ in 'ideal_mhd':
        subdirectory = 'ideal_mhd'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))
    elif __input_case__ in 'ambipolar_diffusion':
        subdirectory = 'ambipolar_diffusion'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))

    nproc = min(len(file_hdf5), cpu_count())

    from tqdm import tqdm
    results = list(tqdm(Pool.imap(return_stats, file_hdf5), total=len(file_hdf5)))

    with Pool(processes=nproc) as pool:
        results = pool.map(return_stats, file_hdf5)

    print("Finished processing all files.")