from library import *
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import os, sys

"""
# Reduction Factor Along Field Lines (R vs S)

"""

#pocket_finder(bmag, cycle=0, plot =False):
import glob

cycle = str(sys.argv[-1])

# percentual error goes commonly into 2% but it can grow up to 6%
# SG has larger discrepancy with original data
# AG smoothes out micro-pockets

origin_folder = 'getLines/430'


radius_vector  = np.array(np.load(os.path.join(origin_folder, f'ArePositions{cycle}.npy'), mmap_mode='r'))
distance       = np.array(np.load(os.path.join(origin_folder, f'ArepoTrajectory{cycle}.npy'), mmap_mode='r'))
bfield         = np.array(np.load(os.path.join(origin_folder, f'ArepoMagneticFields{cycle}.npy'), mmap_mode='r'))
numb_density   = np.array(np.load(os.path.join(origin_folder, f'ArepoNumberDensities{cycle}.npy'), mmap_mode='r'))
volume         = np.array(np.load(os.path.join(origin_folder, f'ArepoVolumes{cycle}.npy'), mmap_mode='r'))
radius = np.linalg.norm(radius_vector, axis=-1)  # Displacement magnitudes

print(" Data Successfully Loaded")


if False: # smoothing of bfield
    """ Smoothing the magnetic field lines """

    ds = np.abs(distance[1:] - distance[:-1])  # Shape (N,)
    ds /= np.sum(ds)
    ds[ds == 0.] = np.min(ds)

    field = bfield[1:]*1.0e+6 # in micro Gauss

    Rs = []
    InvRs = []
    NormBfield = []

    def adaptative_sigma(field, ds, fraction=0.1, polyorder=2):
        N = len(field)

        # Compute local variability using the step size gradient (local_std)
        local_std = np.abs(ds)/np.mean(ds)  # Local variability measure
        local_std2 = np.zeros_like(local_std)  # To store adaptive window sizes
        
        # Normalize local_std to [0,1] for scaling purposes
        local_std = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-5)  # Avoid division by zero
        
        # Define min and max window sizes
        min_window = polyorder + 1  # Minimum window size must be at least polyorder + 1
        max_window = max(min_window, int(fraction * N))  # Max window size is a fraction of the total data length
        
        # Inverse scaling of local_std (smaller std -> smaller window, larger std -> larger window)
        for i in range(N):
            # Inverse relationship: Larger local_std -> larger window
            win_size = int(min_window + 0.5*(1 - local_std[i]) * (max_window - min_window))  # Inverse scaling
            
            if win_size % 2 == 0:  # Ensure window size is odd for Savitzky-Golay filter
                win_size += 1
            local_std2[i] = win_size
                # Plot window size (local_std2)
        plt.plot(local_std/np.max(local_std)*np.max(local_std2),  label="$ds*\sigma^{max}/ds_{mean}$")
        plt.plot(local_std2, label="$\sigma(s)$ Standard deviation")
        plt.xlabel("Index")
        plt.ylabel("Relative Window Size")
        plt.title("Adaptive Window Size Based on Local $\sigma(s)$")
        plt.legend()
        plt.grid()
        plt.savefig(f"./AdaptativeSigma{cycle}.png")
        return None

    # Adaptive Gaussian Filtering (Wider smoothing for noisy regions)
    grad = np.abs(np.gradient(field))
    adaptive_sigma = 5 + 2.5*ds/np.mean(ds) #(ds > np.mean(ds))
    adaptive_gaussian_smoothed = np.array([gaussian_filter1d(field, sigma=s)[i] for i, s in enumerate(adaptive_sigma)]) # adapatative stepsize impact extremes (cell size dependent)

    plt.plot(adaptive_sigma, label="$\sigma(s)$ Standard deviation")
    plt.xlabel("Index")
    plt.ylabel(r"$\sigma(s)$")
    plt.title(r"$\sigma(s) = 3 + 2.5 \frac{dr_{cell}}{dr^{mean}_{cell}}$")
    plt.legend()
    plt.grid()
    plt.savefig(f"./AdaptativeSigma{cycle}.png")

    def reduction(aux):
        pocket, global_info = smooth_pocket_finder(aux, "_c1", plot=False) # this plots
        index_pocket, field_pocket = pocket[0], pocket[1]

        global_max_index = global_info[0]
        global_max_field = global_info[1]

        for i, Bs in enumerate(aux): 
            """  
            R = 1 - \sqrt{1 - B(s)/Bl}
            s = distance traveled inside of the molecular cloud (following field lines)
            Bs= Magnetic Field Strenght at s
            """
            if i < index_pocket[0] or i > index_pocket[-1]: # assuming its monotonously decreasing at s = -infty, infty
                Bl = Bs
            
            p_i = find_insertion_point(index_pocket, i)    
            indexes = index_pocket[p_i-1:p_i+1]       

            if len(indexes) < 2:
                nearby_field = [Bs, Bs]
            else:
                nearby_field = [aux[indexes[0]], aux[indexes[1]]]

            Bl = min(nearby_field)

            if Bs/Bl < 1:
                R = 1 - np.sqrt(1. - Bs/Bl)
            else:
                R = 1

            Rs.append(R)
            InvRs.append(1/R)

        return Rs, InvRs

    # Plot Results
    plt.figure(figsize=(12, 8))
    plt.plot(distance[:-1], field, label='Original', color='grey', alpha=0.6, linewidth=2, linestyle='-')
    plt.plot(distance[:-1], adaptive_gaussian_smoothed, label='Adaptive Gaussian', linestyle='--') # follows too closely the micro-pockets
    #plt.plot(distance[:-1], adaptive_savgol, label='Adaptative Savitzky-Golay', linestyle='--') # maxima is smoothed down 
    plt.legend()
    plt.savefig(f"./ProfileCompareFilters{cycle}.png")

    smoothing_algos = [field, adaptive_gaussian_smoothed]#, adaptive_savgol]#, , fourier_smoothed]
    smoothing_names = ["field", "adaptive_gaussian"]#, "adaptative_savgol"]#, "savgol_smoothed", "fourier_smoothed"]

    diffRs = []
    for o, algo  in enumerate(smoothing_algos):
        
        Rs, InvRs = reduction(algo)
        diffRs.append([np.mean(Rs),np.mean(InvRs)])

    data_Rs = abs(diffRs[0][0])
    data_InvRs = abs(diffRs[0][1])
        
    Rs, InvRs = diffRs[1][0], diffRs[1][1]
    percentage_Rs    = abs(Rs- data_Rs)*100
    percentage_InvRs = abs(InvRs-data_InvRs)*100
    plt.text(
        0.25, 0.75, 
        rf"$\Delta R (AG) = {percentage_Rs:.4f}$, ",
        transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
    )
    plt.text(
        0.25, 0.60, 
        r"$n_g^{threshold} = 10^2$ cm$^{-3}$",
        transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
    )
    plt.xlabel('Path distance (Pc)')
    plt.ylabel('Field Magnitude $\mu$G')
    plt.title('Adaptative Convolution $(B*K)$')
    plt.grid()
    plt.savefig(f"./PocketMeanComparison{cycle}.png")

""" Calculate Forward and Backward Column Densities """

ds = abs(np.diff(distance))

Nplus = np.cumsum(numb_density[1:]*ds)

Nminus = np.cumsum(numb_density[1:][::-1]*ds[::-1])

plt.plot(Nplus)
plt.plot(Nminus)
plt.yscale('log')
plt.show()
exit()


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
C = 2.43e+15            
Enot = 500e+6
Jstar = 2.4e+15*(1.0e+6)**(0.1)/(Enot**2.8)

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6


""" Column Densities N_+(mu, s) & N_-(mu, s)"""

size = distance.shape[0]

mu_ism = np.flip(np.logspace(-3,0,50))

print(size, len(mu_ism))

Nforward  = np.zeros((len(mu_ism), size))+1.0e19

B_ism     = bfield[0]

for i, mui_ism in enumerate(mu_ism):

    for j in range(size):

        print(mui_ism)
        n_g = numb_density[j]
        print(n_g)
        Bsprime = bfield[j]
        print(Bsprime)        
        deno = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        print()
        deno_mu_local = 1 - (bfield[j]/bfield[0])*(1 - mui_ism**2)
        

        if (deno < 0) or (deno_mu_local < 0):
            """ 
            1 - (Bsprime/B_ism) * (1 - muj_ism**2) < 0 
            or 
            mu_local non-existent
            """
            break
        
        delta_nj  = (distance[j]- distance[j-1]) / np.sqrt(deno)
        Nforward[i,j] = Nforward[i,j-1] + n_g*delta_nj

size = distance.shape[0]

mu_ism = np.flip(np.logspace(-1,0,50))

Nbackward  = np.zeros((len(mu_ism), size))+1.0e19

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

mu_ism = np.flip(np.logspace(-1,0,10))

log_forward_ion_rate = np.zeros((len(mu_ism), len(Nforward[0,:])))

for i, mui in enumerate(mu_ism):

    for j, Nj in enumerate(Nforward[i,:]): 
    
        jl_dE = 0.0
    
        for k, E in enumerate(energy): # will go from 3,4,5,6--- almost in integers#
            #print(E,d,Lstar,Estar,Nj)

            Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) # E_i(E, N)

            isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
            llei = loss_function(Ei)           # log_10(L(E_i))
            jl_dE += isms*llei*diff_energy[k] # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        
            if 1 - (bfield[j]/bfield[0])*(1 - mui**2) > 0:
                mu_local = np.sqrt(1 - (bfield[j]/bfield[0])*(1 - mui**2))
                Jacobian = (mui/mu_local)*(bfield[j]/bfield[0])    
            else:
                Jacobian = 0.0
                #break
            Jacobian = 1.0 

        log_zeta_Ni = np.log10(jl_dE / epsilon)  # jacobian * jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon

        if np.isnan(log_zeta_Ni):  # or use numpy.isnan(result) for numpy arrays
            log_zeta_Ni = 0.0

        print(mui,log_zeta_Ni)
        log_forward_ion_rate[i,j] = log_zeta_Ni

not_nan = log_forward_ion_rate[-1,:] != 0
zeros = np.where(log_forward_ion_rate[-1,:] == 0)
print(log_forward_ion_rate.shape)
print(zeros)
print(log_forward_ion_rate[-1,not_nan])

#Plot log(y) vs x
plt.plot(distance[not_nan],log_forward_ion_rate[-1, not_nan], label=r'$\log(y)$ vs $x$', color='b')

# Set x-axis to log scale
plt.xscale('log')

# Labels and title
plt.xlabel('x')
plt.ylabel(r'$\log(y)$')
plt.title('Plot of $\log(y)$ vs x')

# Optional: Adding a grid
plt.grid(True)

# Optional: Adding a legend
plt.legend()

# Show the plot
plt.show()

mu_ism = np.flip(np.logspace(-1,0,10))

#Plot log(y) vs x
plt.plot(mui_ism, label=r'$\log(\mu_i)$', color='b')

# Set x-axis to log scale
plt.yscale('log')

# Labels and title
plt.xlabel('x')
plt.ylabel(r'$\log(\mu_i)$')
plt.title('Plot of $\log(\mu_i)$ vs x')

# Optional: Adding a grid
plt.grid(True)

# Optional: Adding a legend
plt.legend()

# Show the plot
plt.show()