import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import functools
import numpy as np
from src.library import *
import csv, glob, os, sys, time
import h5py

start_time = time.time()

FloatType = np.float128
IntType = np.int32


if __name__ == '__main__':

    data = h5py.File("arepo_data/ideal_mhd/snap_ideal_495.hdf5", 'r')
    MagneticFieldDivergence = np.asarray(data['PartType0']['MagneticFieldDivergence'], dtype=FloatType)
    MagneticFieldDivergenceAlt = np.asarray(data['PartType0']['MagneticFieldDivergenceAlternative'], dtype=FloatType)

    Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
    Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)#*gauss_code_to_gauss_cgs
    Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)#*pc_to_cm


    """

    \int_S \vec{B} \cdot \vec{n} dA

    """
    # sphere or radius 1.0 pc

    center0 = Pos[np.argmax(Density)]

    # center to density peak
    xc = Pos[:, 0] - center0[0]
    yc = Pos[:, 1] - center0[1]
    zc = Pos[:, 2] - center0[2]
    R = Pos[:, :] - center0

    out_r = 0.000100 # size of the clump
    in_r  = 0.000099

    # --- Shell selection ---
    ca = xc*xc + yc*yc + zc*zc < out_r
    cb = xc*xc + yc*yc + zc*zc > in_r
    shell = np.logical_and(ca, cb)
    print("Cells at the edge:", np.sum(shell))

    # --- Normal vectors ---
    mags = np.linalg.norm(R, axis=1)
    nr = R[shell, :] / mags[shell, None]

    # --- Magnetic field on shell ---
    B = Bfield[shell, :]
    Brms = np.sqrt(np.mean(np.sum(B**2, axis=1)))

    # --- Area element per cell ---
    A = 4 * np.pi * ((out_r + in_r)/2)**2
    dA = A / np.sum(shell)

    # --- Magnetic flux (Gauss law for B) ---
    # np.einsum is einstein summation (i.e. dot product)
    BdotdA = np.einsum('ij,ij->i', B, nr) * dA
    Phi_S_CodeUnits = np.sum(BdotdA) # CodeGauss/pc
    Phi_S_CGS = Phi_S_CodeUnits*gauss_code_to_gauss_cgs*pc_to_cm*pc_to_cm
    Phi_S_Const = Brms * A
    ratio = Phi_S_CodeUnits/ Phi_S_Const
    
    #print("[CGSUnits] Total magnetic flux through surface:", Phi_S_CGS)
    print("[CodeUnits]Total magnetic flux through surface:", Phi_S_CodeUnits)
    print("[RMS]      Total magnetic flux (const. field) :", Phi_S_Const)
    print("[Adim]     Ratio RMS flux to calculated:", ratio)

    fig, ax = plt.subplots()
    Bmag = np.linalg.norm(B, axis=1)
    sc = ax.scatter(xc[shell], yc[shell], c=Bmag, cmap='viridis', label="XY")
    plt.colorbar(sc, ax=ax, label='Color Intensity')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Scatter Plot with Color Bar')
    plt.savefig('scatter_plot.png')
    plt.close()

