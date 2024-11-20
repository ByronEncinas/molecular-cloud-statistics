from library import *
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
# Reduction Factor Along Field Lines (R vs S)

"""

#pocket_finder(bmag, cycle=0, plot =False):
import glob

cycle = ''

radius_vector  = np.array(np.load(f"arepo_npys/ArePositions{cycle}.npy", mmap_mode='r'))
distance       = np.array(np.load(f"arepo_npys/ArepoTrajectory{cycle}.npy", mmap_mode='r'))
bfield         = np.array(np.load(f"arepo_npys/ArepoMagneticFields{cycle}.npy", mmap_mode='r'))
numb_density   = np.array(np.load(f"arepo_npys/ArepoNumberDensities{cycle}.npy", mmap_mode='r'))

print(" Data Successfully Loaded")

""" Parameters """

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

# Flux constant (eV^-1 cm^-2 s^-1 sr^-1)
C = 2.43e+15            # Proton in Low Regime (A. Ivlev 2015) https://iopscience.iop.org/article/10.1088/0004-637X/812/2/135
Enot = 500e+6
Jstar = 2.4e+15*(1.0e+6)**(0.1)/(Enot**2.8)

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6


""" Column Densities N_+(mu, s) & N_-(mu, s)"""

size = distance.shape[0]

mu_ism = np.flip(np.logspace(-3,0,50))

Nforward  = np.zeros(len(mu_ism), size)

B_ism     = bfield[0]

for i, mui_ism in enumerate(mu_ism):

    for j in range(size):

        n_g = numb_density[j]
        Bsprime = bfield[j]
        deno = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)

        deno_mu_local = 1 - (bfield[j]/bfield[0])*(1 - mui_ism**2)

        if deno < 0 or deno_mu_local < 0:
            """ 
            1 - (Bsprime/B_ism) * (1 - muj_ism**2) < 0 
            or 
            mu_local non-existent
            """
            break
        
        delta_nj  = (distance[j]- distance[j-1]) / np.sqrt(deno)
        Nforward[i,j] = Nforward[i,j-1] + n_g*delta_nj

size = distance.shape[0]

mu_ism = np.flip(np.logspace(-3,0,50))

Nbackward  = np.zeros(len(mu_ism), size)

rev_numb_density = numb_density[::-1]
rev_bfield        = bfield[::-1]

B_ism      = bfield[-1]

for i, mui_ism in enumerate(mu_ism):

    for j in range(size):

        n_g = rev_numb_density[j]
        Bsprime = rev_bfield[j]
        deno = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)

        deno_mu_local = 1 - (bfield[j]/bfield[0])*(1 - mui_ism**2)

        if deno < 0 or deno_mu_local < 0:
            """ 
            1 - (Bsprime/B_ism) * (1 - muj_ism**2) < 0 
            or 
            mu_local non-existent
            """
            break
        
        delta_nj  = (distance[j]- distance[j-1]) / np.sqrt(deno)
        Nbackward[i,j] = Nbackward[i,j-1] + n_g*delta_nj


""" Ionization Rate for N = N_+(s,mu) """

energy = np.array([1.0e+2*(10)**(14*k/size) for k in range(size)])  # Energy values from 1 to 10^15
diff_energy = np.array([energy[k]-energy[k-1] for k in range(len(energy))])
diff_energy[0] = energy[0]

ism_spectrum = lambda x: Jstar*(x/Estar)**a
loss_function = lambda z: Lstar*(Estar/z)**d

log_forward_ion_rate = np.zeros((len(mu_ism), len(Nforward[0,:])))

for i, mui in enumerate(mu_ism):

    for j, Nj in enumerate(Nforward[i,:]): 
    
        jl_dE = 0.0
    
        for k, E in enumerate(energy): # will go from 3,4,5,6--- almost in integers#

            Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) # E_i(E, N)

            isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
            llei = loss_function(Ei)           # log_10(L(E_i))
            jl_dE += isms*llei*diff_energy[k] # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
            if 1 - (bfield[j]/bfield[0])*(1 - mui**2) > 0:
                mu_local = np.sqrt(1 - (bfield[j]/bfield[0])*(1 - mui**2))
                Jacobian = (mui/mu_local)*(bfield[j]/bfield[0])    
            else:
                Jacobian = 0.0
                break
            Jacobian = 1.0 

        zeta_Ni = jl_dE / epsilon  # jacobian * jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon

        log_forward_ion_rate[i,j] = np.log10(zeta_Ni)