from library import *
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import os, sys
import glob

""" Parameters """

size = 1000

Ei = 1.0e+0
Ef = 1.0e+15

N0 = 10e+19
Nf = 10e+27

n0 = 150 #cm−3 and 
k  = 0.5 # –0.7

d = 0.82
a = 0.1 # spectral index either 0.1 from Low Model, or \in [0.5, 2.0] according to free streaming analytical solution.

# mean energy lost per ionization event
epsilon = 35.14620437477293

# Luminosity per unit volume for cosmic rays (eV cm^2)
Lstar = 1.4e-14

# Flux constant (eV^-1 cm^-2 s^-1 sr^-1) C*(10e+6)**(0.1)/(Enot+6**2.8)
Jstar = 2.43e+15*(10e+6)**(0.1)/(500e+6**2.8) # Proton in Low Regime (A. Ivlev 2015) https://iopscience.iop.org/article/10.1088/0004-637X/812/2/135

# Flux constant (eV^-1 cm^-2 s^-1)
C = 2.43e+15*4*np.pi
Enot = 500e+6
Jstar = 2.4e+15*(1.0e+6)**(0.1)/(Enot**2.8)

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6

logE0, logEf = 6, 9
energy = np.logspace(logE0, logEf, size)
print("energy range for CR spectrum (log10): ", logE0, logEf)
diff_energy = np.array([energy[k]-energy[k-1] for k in range(len(energy))])
diff_energy[0] = energy[0]

ism_spectrum = lambda x: C*(x**0.1/((Estar + x)**2.8))
ism_spectrum = lambda x: Jstar*(x/Estar)**a
loss_function = lambda z: Lstar*(Estar/z)**d

Neff = np.logspace(19, 26, size)

if True:
    cross_data = np.load('arepo_data/cross_pH2_rel_1e18.npz')
    loss_data  = np.load('arepo_data/Kedron_pLoss.npz')

    cross = interpolate.interp1d( cross_data["E"], cross_data["sigmap"])
    loss = interpolate.interp1d(loss_data["E"], loss_data["L_full"])


if True: # Padovani & Alexei Models for CR Ionization-rate
    """  
    \mathcal{L} & \mathcal{H} Model: Protons
    """
    
    model_H = [1.001098610761e7, -4.231294690194e6,  7.921914432011e5,
            -8.623677095423e4,  6.015889127529e3, -2.789238383353e2,
            8.595814402406e0, -1.698029737474e-1, 1.951179287567e-3,
            -9.937499546711e-6
    ]


    model_L = [-3.331056497233e+6,  1.207744586503e+6,-1.913914106234e+5,
                1.731822350618e+4, -9.790557206178e+2, 3.543830893824e+1, 
            -8.034869454520e-1,  1.048808593086e-2,-6.188760100997e-5, 
                3.122820990797e-8]

    logzl = []
    for i,Ni in enumerate(Neff):
        lzl = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
        logzl.append(lzl)

    logzh = []

    for i,Ni in enumerate(Neff):
        lzh = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_H)] )
        logzh.append(lzh)

    from scipy import interpolate

    log_L = interpolate.interp1d(Neff, logzl)
    log_H = interpolate.interp1d(Neff, logzh)
    plt.plot(Neff, log_H(Neff), label = 'Model H')
    plt.plot(Neff, log_L(Neff), label = 'Model L')
    plt.xscale('log')
    plt.legend()
    plt.savefig('./images/i_rate/padovani2018')
    plt.close()

if True:
    line = int(sys.argv[-1])

    import h5py
    import numpy as np

    h5_path = "./stats/ideal/430/DataBundle.h5"

    with h5py.File(h5_path, "r") as f:

        # --- Datasets ---
        x_input         = f["starting_point"][()]
        n_rs            = f["number_densities"][()]
        B_rs            = f["magnetic_fields"][()]
        c_rs            = f["column_path"][()]
        reduction_fac   = f["reduction_factor"][()]
        directions      = f["directions"][()]
        average_column  = f["column_los"][()]
        numb_density    = f["densities"][()][:, line]
        magnetic_field =  f["bfields"][()][:, line]
        radius_vector  =  f["vectors"][()][:, line, :]

        # --- Metadata / attributes ---
        cores_used     = f.attrs["cores_used"]
        pre_alloc_num  = f.attrs["pre_allocation_number"]
        rloc           = f.attrs["rloc"]
        max_cycles     = f.attrs["max_cycles"]
        center         = f.attrs["center"]
        volume_range   = f.attrs["volume_range"]
        density_range  = f.attrs["density_range"]


    # --- Adjustments / Resizing ---
    mask = np.where(numb_density > 0.0)[0]
    start, end = mask[0], mask[-1]
    follow_index = numb_density.shape[0]

    print(start, follow_index//2, end)
    numb_density    = numb_density[start:end+1]
    magnetic_field =  magnetic_field[start:end+1]
    radius_vector  =  radius_vector[start:end+1, :]*pc_to_cm
    trajectory = np.cumsum(np.linalg.norm(radius_vector, axis=1)) #np.insert(, 0, 0.0)

    print("trajectory:           ", trajectory.shape)
    print("radius_vectors:       ", radius_vector.shape)
    print("numb_density:         ", numb_density.shape)
    print("magnetic_field:       ", magnetic_field.shape)

    am = np.argmax(magnetic_field)
    numb_density    = numb_density[:am]
    magnetic_field =  magnetic_field[:am]
    radius_vector  =  radius_vector[:am, :]
    trajectory = np.cumsum(np.linalg.norm(radius_vector, axis=1))

    fig, axs = plt.subplot_mosaic([["a"], ["b"], ["c"]], figsize=(10, 8), layout="constrained")
    axs["a"].plot(trajectory, c="b"); axs["a"].set_title("Trajectory"); axs["a"].set_ylabel("Trajectory")
    axs["b"].plot(numb_density, c="g"); axs["b"].set_title("Number Density"); axs["b"].set_ylabel("n (cm$^{-3}$)")
    axs["c"].plot(magnetic_field, c="r"); axs["c"].set_title("Magnetic Field"); axs["c"].set_ylabel("B (μG)"); axs["c"].set_xlabel("Index")
    plt.savefig("./images/i_rate/mosaic_plot.png"); plt.close()
    
""" Column Densities N_+(mu, s) & N_-(mu, s)"""

mu_ism = np.logspace(-1, 0, num=50) #np.array([1.0])#

def column_density(radius_vector, magnetic_field, numb_density, direction=''):
    trajectory = np.cumsum(np.linalg.norm(radius_vector, axis=1)) #np.insert(, 0, 0.0)
    dmui = np.insert(np.diff(mu_ism), 0, mu_ism[0])    
    ds = np.insert(np.linalg.norm(np.diff(radius_vector, axis=0), axis=1), 0, 0.0)
    Nmu  = np.zeros((len(magnetic_field), len(mu_ism)))
    dmu = np.zeros((len(magnetic_field), len(mu_ism)))
    mu_local = np.zeros((len(magnetic_field), len(mu_ism)))
    B_ism     = magnetic_field[0]

    for i, mui_ism in enumerate(mu_ism):

        for j in range(len(magnetic_field)):

            n_g = numb_density[j]
            Bsprime = magnetic_field[j]
            mu_local2 = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
            
            if (mu_local2 <= 0):
                break
            mu_local[j,i] = np.sqrt(mu_local2)
            dmu[j,i] = (mui_ism/mu_local[j,i])*(Bsprime/B_ism)*dmui[i]
            Nmu[j, i] = Nmu[j - 1, i] + n_g * ds[j] / mu_local[j, i] if j > 0 else n_g * ds[j] / mu_local[j, i]
    fig, ax = plt.subplots()

    for mui in range(Nmu.shape[1]):
        ax.plot(mu_local[:, mui], label=fr"$\mu_i = {mu_ism[mui]}$")

    ax.set_title(r"Evolution of $cos(\alpha(\alpha_i), s)$ along trajectory")
    ax.set_xlabel("Index (Visualization Graph)")
    ax.set_ylabel(r"$\cos(\alpha(\alpha_i), s)$")
    ax.set_xscale('log')
    #ax.legend()
    fig.tight_layout()

    plt.savefig(f'./images/i_rate/mu_local_{direction}.png')
    plt.close()

    return Nmu, mu_local, dmu, trajectory

Npmu, mu_local_fwd, dmu_fwd, t_fwd = column_density(radius_vector, magnetic_field, numb_density, "fwd")

Nmmu, mu_local_bwd, dmu_bwd, t_bwd = column_density(radius_vector[::-1, :], magnetic_field[::-1], numb_density[::-1], "bwd")

if True:

    for mui in range(Npmu.shape[1]):
        plt.plot(Npmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}")    
        #plt.legend()

    plt.title("Column density $N_+(\mu_i)$  ")
    plt.xlabel("Index")
    plt.ylabel(r"$N_+(\mu_i, s)$")
    #plt.xscale('log')
    plt.yscale('log')
    plt.savefig('./images/i_rate/Nfwd.png')  # Save as PNG with high resolution
    plt.close()

    for mui in range(Npmu.shape[1]):
        plt.plot(Nmmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}")    
        #plt.legend()

    plt.title("Column density $N_-(\mu_i)$  ")
    plt.xlabel("Index")
    plt.ylabel(r"$N_-(\mu_i, s)$")
    #plt.xscale('log')
    plt.yscale('log')
    plt.savefig('./images/i_rate/Nbwd.png')  # Save as PNG with high resolution
    plt.close()

    for mui in range(dmu_fwd.shape[1]):
        plt.plot(dmu_fwd[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
        #plt.legend()

    plt.title(r"Evolution of $\frac{B}{B_i}\frac{\mu_i}{\mu}$ (Forwards)")
    plt.xlabel("trajectory (cm)")
    plt.ylabel(r"$\frac{\partial \mu_i}{\partial \mu}$")
    plt.xscale('log')
    plt.savefig('./images/i_rate/Jfwd.png')  # Save as PNG with high resolution
    plt.close()
    
    for mui in range(dmu_bwd.shape[1]):
        plt.plot(dmu_bwd[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
        #plt.legend()

    plt.title(r"Evolution of $\frac{B}{B_i}\frac{\mu_i}{\mu}$ (Backwards)")
    plt.xlabel("trajectory (cm)")
    plt.ylabel(r"$\frac{\partial \mu_i}{\partial \mu}$")
    plt.xscale('log')
    plt.savefig('./images/i_rate/Jbwd.png')  # Save as PNG with high resolution
    plt.close()

"""
N_mirr = N_+ + int_{s'}

"""

def mirrored_column_density(radius_vector, magnetic_field, numb_density, Nmu, direction=''): 

    ds    = np.insert(np.linalg.norm(np.diff(radius_vector, axis=0), axis=1), 0, 0.0)
    Nmir  = np.zeros((len(magnetic_field), len(mu_ism)))
    s_max = np.argmax(magnetic_field) + 1
    B_ism = magnetic_field[0]

    for i, mui_ism in enumerate(mu_ism): # cosina alpha_i
        for s in range(s_max):            # at s
            N = Nmu[s, i]
            for s_prime in range(s_max-s): # get mirrored at s_mirr; all subsequent points s < s_mirr up until s_max
                # sprime is the integration variable.
                if (magnetic_field[s_prime] > B_ism*(1-mui_ism**2)):
                    break
                mu_local = np.sqrt(1 - magnetic_field[s_prime]*(1-mui_ism**2)/B_ism )
                s_mir = s + s_prime
                dens  = numb_density[s:s_mir]
                diffs = ds[s:s_mir] 
                N += np.sum(dens*diffs/mu_local)
            Nmir[s,i] = N

    for i, mu in enumerate(mu_ism):
        plt.plot(Nmir[:, i], label=fr'$\mu_{{\mathrm{{ISM}}}}={mu:.2f}$')
    
    plt.title("Column density $N_{mirr}^+(s)$")
    plt.xlabel("Steps")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'./images/i_rate/N{direction}.png')
    plt.close()

    return Nmir

max_arg = np.argmax(magnetic_field)

Nmir_fwd = mirrored_column_density(radius_vector[:max_arg,:], magnetic_field[:max_arg], numb_density[:max_arg], Npmu[:max_arg,:], 'mir_fwd')
Nmir_bwd = mirrored_column_density(radius_vector[max_arg+1:,:][::-1,:], magnetic_field[max_arg+1:][::-1], numb_density[max_arg+1:][::-1], Nmmu[max_arg+1:,:], 'mir_bwd')

""" Ionization Rate for N = N(s) """

def ionization_rate(Nmu, mu_local, dmu, direction = ''):
    zeta_mui = np.zeros_like(Nmu)
    zeta = np.zeros_like(Nmu[:,0])

    for l, mui in enumerate(mu_ism):

        for j, Nj in enumerate(Nmu[:,l]):
            mu_ = mu_local[j,l]
            if mu_ <= 0:
                break

            jl_dE = 0.0

            #  Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) 
            Ei = ((energy)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj/mu_)**(1 / (1 + d))

            isms = ism_spectrum(Ei)                        # log_10(j_i(E_i))
            llei = loss_function(Ei)                       # log_10(L(E_i))
            sigma_ion = cross(energy)
            spectra   = 0.5*isms*llei/loss_function(energy)  
            jl_dE = np.sum(isms*llei*diff_energy)  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
            zeta_mui[j, l] = jl_dE / epsilon 
            #zeta_mui[j, l] = np.sum(spectra*sigma_ion*diff_energy)

    zeta = np.sum(dmu * zeta_mui, axis = 1)

    plt.plot(zeta_mui, label=f'$\zeta_-(s) $', alpha = 0.5)
    plt.title(r"$\zeta(N, \mu)$", fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'./images/i_rate/zeta_{direction}.png')
    plt.close()

    return zeta, zeta_mui

#zeta_fwd, zeta_mui_fwd  = ionization_rate(Npmu, mu_local_fwd, dmu_fwd, 'fwd')
#zeta_bwd, zeta_mui_bwd = ionization_rate(Nmmu, mu_local_bwd, dmu_bwd,'bwd')

zeta_mir_fwd,  zeta_mui_mir_fwd  = ionization_rate(Nmir_fwd, mu_local_fwd[:max_arg,:], dmu_fwd[:max_arg,:], 'mir_fwd')
zeta_mir_bwd, zeta_mui_mir_bwd = ionization_rate(Nmir_bwd, mu_local_bwd[max_arg+1:,:], dmu_bwd[max_arg+1:,:], 'mir_bwd')

fig, ax = plt.subplots()
Nmir = np.sum(np.concatenate((Nmir_fwd, Nmir_bwd[::-1]), axis=0), axis=1)
zeta = np.concatenate((zeta_mir_fwd, zeta_mir_bwd[::-1]), axis=0)

ax.scatter(Nmir, zeta, s=1, marker='x',color="red",label=fr"$\zeta$")
#ax.plot(Nmir,label=fr"$\zeta$")
ax.plot(Neff, 10**log_H(Neff), label = 'Model H')
ax.plot(Neff, 10**log_L(Neff), label = 'Model L')
ax.set_title(r"Total Ionization")
ax.set_xlabel("Distance (cm)")
ax.set_ylabel(r"$\zeta(s)$")
ax.set_yscale('log')
ax.set_xscale('log')

#ax.legend()
fig.tight_layout()

plt.savefig(f'./images/i_rate/total_zeta.png')
plt.close()

