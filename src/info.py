import h5py
import numpy as np
import pprint

# yes, chatgpt helped extract this code, sue me. I lost the one I had made previously.

def explore_snapshot_config(filename):
    with h5py.File(filename, 'r') as f:
        print("🔹 Snapshot file:", filename)
        print("\nTop-level groups:")
        for key in f.keys():
            print(" ", key)

        print("\n Header attributes:")
        header = f['Header'].attrs
        for k in header:
            print(f"  {k}: {header[k]}")

        # Try to extract unit info (if present)
        if 'Units' in f:
            print("\n Units:")
            for k in f['Units'].attrs:
                print(f"  {k}: {f['Units'].attrs[k]}")
        else:
            print("\n Units: Not found in file")

        # Particle types
        print("\n Particle Types and Fields:")
        for key in f.keys():
            if key.startswith('PartType'):
                print(f"  {key}:")
                for dataset in f[key].keys():
                    shape = f[key][dataset].shape
                    dtype = f[key][dataset].dtype
                    print(f"    - {dataset}: shape={shape}, dtype={dtype}")

        # Other possible metadata groups
        for key in f.keys():
            if key not in ['Header', 'Units'] and not key.startswith('PartType'):
                print(f"\n Additional group: {key}")
                try:
                    for subkey in f[key].keys():
                        print(f"  - {subkey}")
                except AttributeError:
                    pass  # not a group


import h5py
import json
from pprint import pprint

def extract_simulation_config(filename):
    with h5py.File(filename, 'r') as f:
        print(f"\n Reading: {filename}")

        # -------- Header --------
        print("\n HEADER attributes:")
        if 'Header' in f:
            hdr = f['Header'].attrs
            header_data = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in hdr.items()}
            pprint(header_data)
        else:
            print("No 'Header' group found.")

        # -------- Parameters --------
        print("\n PARAMETERS (AREPO config options):")
        if 'Parameters' in f:
            params = f['Parameters'].attrs
            param_data = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in params.items()}
            pprint(param_data)
        else:
            print("No 'Parameters' group found.")

        # -------- Config --------
        print("\n🔧 CONFIG (compile-time flags / physics modules):")
        if 'Config' in f:
            config = f['Config'].attrs
            config_data = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in config.items()}
            pprint(config_data)
        else:
            print("No 'Config' group found.")


PARAMETERS : dict = {
    'CourantFac': np.float64(0.3),
     'ErrTolIntAccuracy': np.float64(0.012),
 'InitGasTemp': np.float64(4500.0),
 'JeansNumber': np.float64(32.0), 
 'MaxSizeTimestep': np.float64(0.1),
 'MaxTimebinSpread': np.int32(8),
 'MaxVolume': np.float64(128.0),
  'MinSizeTimestep': np.float64(1e-20),
  'TimeBegin': np.float64(0.0),
  'TimeBetSnapshot': np.float64(5e-07),
  'TimeMax': np.float64(12.8),
  'TimeOfFirstSnapshot': np.float64(0.0),
 'TypeOfTimestepCriterion': np.int32(0),
 'WaitingTimeFactor': np.float64(1.0)}


# Run on your snapshot file:
import glob
s = glob.glob(f"arepo_data/ideal_mhd/*000.hdf5")

extract_simulation_config(s[0])
explore_snapshot_config(s[0])