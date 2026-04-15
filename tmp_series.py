import glob, os, sys, time, h5py, gc, ctypes
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import warnings
import src.tmp_library as tmplib
from mpi4py import MPI          # Move to TOP of __main__
import pickle
import asyncio
import glob

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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        clst, dlst, tlst, slst, file_hdf5 = tmplib.match_files_to_data(tmplib.__input_case__,__start_snap__)
        _id_ = str(input_file.split('.')[0][0] + input_file.split('.')[0][-1])
        if tmplib.FLAG1 in sys.argv:
            _ia_ = str(input_file.split('/')[1][0]) #ideal/amb
            _id_ = _ia_ + "e-" + str(input_file.split("e-")[1][0])
        print("ID of series.py run is", _id_)
    else:
        clst = dlst = tlst = slst = file_hdf5 = None
        _id_ = None

    clst, dlst, tlst, slst, file_hdf5, _id_ = comm.bcast(
        (clst, dlst, tlst, slst, file_hdf5, _id_), root=0
    )

    assert len(file_hdf5) == len(clst) == len(tlst), "Arrays must all have the same length"

    survivors_fraction = np.zeros(file_hdf5.shape[0])

    print(_id_, file_hdf5)

    df_stats = dict()
    df_fields= dict()

    for each in range(rank, len(file_hdf5), size):
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


        if x_input is None:
            print(f"[Snap] snap {tmplib.snap}: skipping", flush=True)
            tmplib.config_arepo(filename, center, True)
            continue

        directions=tmplib.fibonacci_sphere(20)        # dodecahedron (12 faces), and icosahedron (20 faces)
        try:
            print(tmplib.__threshold__)
            __0, __1, mean_column, median_column = tmplib.line_of_sight(x_init=x_input, directions=directions, n_crit=tmplib.__threshold__)
        except Exception as e:
            warnings.warn(f"[snap={tmplib.snap}]", e)
            print(f"[LOS] Invalid result from intergration: {tmplib.snap}: skipping", flush=True)
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
        print("__used_slots__ : ",__0.shape, flush=True)

        if np.log10(tmplib.__threshold__) < 2: 
            r_u, n_rs, B_rs, survivors2 = tmplib.eval_reduction(magnetic_fields, numb_densities, follow_index, tmplib.__threshold__*10)
            r_l, _1, _2, _3 = tmplib.eval_reduction(magnetic_fields, numb_densities, follow_index, tmplib.__threshold__)
        else:
            r_u, n_rs, B_rs, survivors2 = tmplib.eval_reduction(magnetic_fields, numb_densities, follow_index, tmplib.__threshold__)
            r_l, _1, _2, _3 = tmplib.eval_reduction(magnetic_fields, numb_densities, follow_index, tmplib.__threshold__) # make redundant
        
        survivors = np.logical_and(survivors1, survivors2)

        print(np.sum(survivors)/survivors.shape[0], " Survivor fraction", flush=True)

        survivors_fraction[each] = np.sum(survivors)/survivors.shape[0]
        u_input         = x_input[np.logical_not(survivors),:] # pc
        x_input         = x_input[survivors,:]                 # pc
        radius_vectors  = radius_vectors[:, survivors, :]      # pc
        numb_densities  = numb_densities[:, survivors]         # cm-3
        magnetic_fields = magnetic_fields[:, survivors]*tmplib.gauss_code_to_gauss_cgs # Gauss CGS
        mean_column     = mean_column[survivors]               # cm-2
        median_column   = median_column[survivors]             # cm-2
        path_column     = path_column[survivors]               # cm-2
        n_rs            = n_rs[survivors]                      # cm-3
        B_rs            = B_rs[survivors]*tmplib.gauss_code_to_gauss_cgs # Gauss CGS
        r_u             = r_u[survivors]                       # Adim
        r_l             = r_l[survivors]                       # Adim
        

        mean_r_u, median_r_u, skew_r_u, kurt_r_u = tmplib.describes(r_u)
        mean_r_l, median_r_l, skew_r_l, kurt_r_l = tmplib.describes(r_l)

        stats_dict = {
            "time": _time, 
            "x_input": x_input,
            "n_rs": n_rs,
            "B_rs": B_rs,
            "n_path": path_column,
            "n_los0": mean_column,
            "n_los1": median_column,
            "surv_fraction": survivors_fraction[each],
            "r_u": r_u,
            "r_l": r_l
        }

        df_stats[str(tmplib.snap)]  = stats_dict
        print(df_stats)
        
        if tmplib.FLAG0 in sys.argv:
            field_dict = {
                "time": _time,
                "directions": directions,
                "x_input": x_input,
                "B_s": magnetic_fields,
                "r_s": radius_vectors,
                "n_s": numb_densities
            }

            df_fields[str(tmplib.snap)]  = field_dict
        if 'Pos' in globals():
            print("\nPos is global", flush=True)

        tmplib.get_globals_memory()            
        # at the end of the loop, drop all that will be reasigned, to avoid memory overflow
        del tree, __0, __1, _1, _2, _3
        del radius_vectors, magnetic_fields, numb_densities

        gc.collect()
        tmplib.config_arepo(filename, center, True)
        tmplib.get_globals_memory()

    with open(f'./series/tmp_{_id_}_rank{rank}.pkl', 'wb') as f:
        pickle.dump(df_stats, f)
        f.flush()
        os.fsync(f.fileno())

    comm.Barrier()
    if rank == 0:
        expected = comm.Get_size()
        asyncio.run(tmplib.merge_and_save(_id_, tmplib.__dense_cloud__))

    elapsed_time =time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"Elapsed time: {hours}h {minutes}m {seconds}s")
    
