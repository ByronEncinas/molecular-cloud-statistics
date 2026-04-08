from src.library import *
import pandas as pd
import numpy as np
import os, glob, sys

def imporfromfile(file, identifier):

    df = pd.read_pickle(file)
    df.index.name = 'snapshot'
    df.index = df.index.astype(int)
    df = df.sort_index()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    
    if df.empty:
        return False
        
    t     = df["time"].to_numpy()
    _ = np.argsort(t)

    t     = df["time"].to_numpy()[_]
    x     = df["x_input"].to_numpy()[_]
    n     = df["n_rs"].to_numpy()[_]
    B     = df["B_rs"].to_numpy()[_]
    Nlos0 = df["n_los0"].to_numpy() [_] # mean
    Nlos1 = df["n_los1"].to_numpy() [_] # median
    Ncrs  = df["n_path"].to_numpy()[_]
    factu = df["r_u"].to_numpy()[_]
    factl = df["r_l"].to_numpy()[_]
    surf  = df["surv_fraction"].to_numpy()[_]
    rurl = [np.mean(ru-rl) for (ru,rl) in zip(factu, factl)]

    radius = np.ceil(np.max([np.max(ris) for ris in x])*100)/100

    if identifier[0] == '6'  and identifier[-1] == '4':
        radius = np.ceil(np.max([np.max(ris) for ris in x])*100)/100
    elif identifier == '4i3' or identifier == '4a3':
        radius = 0.1
    elif identifier == '2i2' or identifier == '2a2':
        radius = 0.5
    elif identifier == '2i1' or identifier == '2a1':
        radius = 0.2
    elif identifier == '2i0' or identifier == '2a0':
        radius = 0.1
        
    if "e-" in identifier:
        int_extract = float(identifier.split("e-")[-1][0])
        radius = 10**int_extract

    return {
        "id": identifier,
        "rloc": radius,
        "t":     t,
        "x":     x,
        "n":     n,
        "B":     B,
        "Nlos0": Nlos0,
        "Nlos1": Nlos1,
        "Ncrs":  Ncrs,
        "factu": factu,
        "factl": factl,
        "surf":  surf,
        "rurl":  rurl
    }

files = sorted(glob.glob(f'./series/pickles/*.pkl'))

df = pd.DataFrame()  # empty DataFrame

for index, file in enumerate(files):
    ID = file.split("/")[-1].split(".")[0][-3:]
    #print(file.split("/")[-1].split(".")[0][-3:])
    data = imporfromfile(file, ID)
    if not data:
        continue
    if data["x"].shape[0] < 5:
        # should I delete? 
        # os.remove(file)
        continue

    info = {
        "identifier": ID,
        "times"  : data["t"].shape[0],
        "dense_cloud"  : f"1.0e+{ID[0]}",
        "rloc"  : data["rloc"],
        "min_size": np.min([ru.shape[0] for ru in data["factu"]]),
        "frac_<1": np.mean(([np.sum(ru[ru<1])/ru.shape for ru in data["factu"]])),
        "rej_lines":  [round(1-np.min(data["surf"]), 4),round(1-np.max(data["surf"]),4)]
    }
    
    if index == 0:
        df = pd.DataFrame([info])
    else:
        df = pd.concat([df, pd.DataFrame([info])], ignore_index=True)

print(df)


