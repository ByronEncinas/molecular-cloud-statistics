from library import *
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import glob

print()


origin_folder = 'arepo_npys/stepsizetest/0.2/'



file, line = sys.argv[1], sys.argv[2]

origin_folder = f'stats/ideal/{file}'

origin_folder = 'getLines/ideal/430'

if False:
    #stats/ideal/430/ColumnDensity6.npy
    #./los_stats/N/ideal/430/', f'eff_column_densities_{line}.npy'
    columns = []
    for line in range(50):

        column_density  = np.array(np.load(f'los_stats/N/ideal/430/ColumnDensities{line}.npy'))*pc_to_cm
        number_density  = np.array(np.load(f'los_stats/N/ideal/430/NumberDensities{line}.npy'))
        position_vectors  = np.array(np.load(f'los_stats/N/ideal/430/Positions{line}.npy'))

        non_zero = np.logical_not(number_density == 0)

        column_density  = column_density[non_zero]
        number_density  = number_density[non_zero]
        position_vectors  = position_vectors[non_zero]
        #print(column_density.shape, number_density.shape, position_vectors.shape)

        columns += [column_density[-1]]
        print(np.mean(columns))


    plt.plot(columns)
    plt.yscale('log')
    plt.show()


    plt.plot(number_density)
    plt.yscale('log')
    plt.show()


    plt.plot(position_vectors[:,0])
    plt.show()

    exit()

radius_vector  = np.array(np.load(os.path.join(origin_folder, f'Positions{line}.npy'), mmap_mode='r'))
distance       = np.array(np.load(os.path.join(origin_folder, f'Trajectory{line}.npy'), mmap_mode='r'))
bfield         = np.array(np.load(os.path.join(origin_folder, f'MagneticFields{line}.npy'), mmap_mode='r'))
numb_density   = np.array(np.load(os.path.join(origin_folder, f'NumberDensities{line}.npy'), mmap_mode='r'))
radius         = np.linalg.norm(radius_vector, axis=-1)

unique_values, counts = np.unique(radius_vector[:,0], return_counts=True)
duplicates = unique_values[counts > 1]

# Iterate over the duplicate values and find their indices
for duplicate in duplicates:
    indices = np.where(radius_vector[:,0] == duplicate)[0]
    print(indices)

if True:
    inter = (abs(np.roll(distance, 1) - distance) != 0) # removing pivot point
    distance = distance[inter]
    ds = np.abs(np.diff(distance, n=1))
    dsm = np.linalg.norm(np.diff(radius_vector[::-1], axis=0), axis=1)    
    distance = distance[:-1]    # Remove the last element of distance
    radius_vector = radius_vector[inter][:-1]
    numb_density  = numb_density[inter][:-1]
    bfield        = bfield[inter][:-1]

    if False:
        Rs = []
        InvRs = []
        NormBfield = []

        plt.plot(adaptive_sigma, label="$\sigma(s)$ Standard deviation")
        plt.xlabel("Index")
        plt.ylabel(r"$\sigma(s)$")
        plt.title(r"$\sigma(s) = 3.0 \frac{dr_{cell}}{dr^{mean}_{cell}}$")
        plt.legend()
        plt.grid()
        plt.savefig(f"./AdaptativeSigma{cycle}.png")

        def reduction(aux):
            pocket, global_info = smooth_pocket_finder(aux, "_c1", plot=False) # this plots
            index_pocket, field_pocket = pocket[0], pocket[1]

            global_max_index = global_info[0]
            global_max_field = global_info[1]

            for i, Bs in enumerate(aux): 
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
        plt.plot(field, label='Original', color='grey', alpha=0.6, linewidth=2, linestyle='-')
        plt.plot(adaptive_gaussian_smoothed, label='Adaptive Gaussian', linestyle='--') # follows too closely the micro-pockets
        plt.legend()

        smoothing_algos = [field, adaptive_gaussian_smoothed]
        smoothing_names = ["field", "adaptive_gaussian"]

        diffRs = []
        for o, algo  in enumerate(smoothing_algos):
            
            Rs, InvRs = reduction(algo)
            diffRs.append([np.mean(Rs),np.mean(InvRs)])

        data_Rs = abs(diffRs[0][0])
        data_InvRs = abs(diffRs[0][1])
            
        Rs, InvRs = diffRs[1][0], diffRs[1][1]
        percentage_Rs    = abs((Rs- data_Rs)/data_Rs)*100
        percentage_InvRs = abs((InvRs-data_InvRs)/data_InvRs)*100
        plt.text(
            0.25, 0.75, 
            rf"$\Delta R (AG) = {percentage_Rs:.4f}$ %",
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


if False:
    for mui in range(Npmu.shape[1]):
        plt.plot(distance, mu_local_plus[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
        plt.legend()

    plt.title(r"Evolution of $cos(\alpha(\alpha_i), s)$ along trajectory")
    plt.xlabel("Distance (pc)")
    plt.ylabel(r"$cos(\alpha(\alpha_i), s)$")
    plt.xscale('log')
    plt.savefig('images/mulocplus.png')  # Save as PNG with high resolution
    plt.close()
    for mui in range(Npmu.shape[1]):
        plt.plot(distance, Npmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}")    
        plt.legend()

    plt.title("Column density $N_+(\mu_i)$  ")
    plt.xlabel("Distance (pc)")
    plt.ylabel(r"$N_+(\mu_i, s)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('images/Nplus.png')  # Save as PNG with high resolution
    plt.close()
    for mui in range(Jacobpmu.shape[1]):
        plt.plot(distance, Jacobpmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
        plt.legend()

    plt.title(r"Evolution of $\frac{B}{B_i}\frac{\mu_i}{\mu}$ along trajectory")
    plt.xlabel("Distance (pc)")
    plt.ylabel(r"$\frac{\partial \mu_i}{\partial \mu}$")
    plt.xscale('log')
    plt.savefig('images/Jplus.png')  # Save as PNG with high resolution
    plt.close()

""" Parameters """

size = ds.shape[0]

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
C = 2.43e+15*4*np.pi            
Enot = 500e+6
Jstar = 2.4e+15*(1.0e+6)**(0.1)/(Enot**2.8)

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6

energy = np.logspace(2, 15, size)
diff_energy = np.array([energy[k]-energy[k-1] for k in range(len(energy))])
diff_energy[0] = energy[0]

ism_spectrum = lambda x: C*(x**0.1/((Estar + x)**2.8))
ism_spectrum = lambda x: Jstar*(x/Estar)**a
loss_function = lambda z: Lstar*(Estar/z)**d

Neff = np.logspace(19, 27, size)

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
"""
for i,Ni in enumerate(Neff):
    lzl = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
    logzl.append(lzl)
"""
#print("Extremes of fitting zeta(N) ", logzl[0], logzl[-1])

#logzetalfit = np.array(logzl)

#logzh = []

"""for i,Ni in enumerate(Neff):
    lzh = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_H)] )
    logzh.append(lzh)
"""
#print("Extremes of fitting zeta(N) ", logzh[0], logzh[-1])

#logzetahfit = np.array(logzh)

#svnteen = np.ones_like(Neff)*(-17)

zeta = np.zeros_like(Neff)

""" Column Densities N_+(mu, s) & N_-(mu, s)"""

mu_ism = np.logspace(-1, 0, num=50) #np.array([1.0])#
dmui = np.insert(np.diff(mu_ism), 0, mu_ism[0])

Npmu  = np.zeros((size, len(mu_ism)))
dmu_plus = np.zeros((size, len(mu_ism)))
mu_local_plus = np.zeros((size, len(mu_ism)))
B_ism     = bfield[0]

for i, mui_ism in enumerate(mu_ism):

    for j in range(size):

        n_g = numb_density[j]
        Bsprime = bfield[j]
        mu_local2 = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        
        if (mu_local2 <= 0):
            break
        
        mu_local_plus[j,i] = np.sqrt(mu_local2)

        dmu_plus[j,i] = (mui_ism/mu_local_plus[j,i])*(Bsprime/B_ism)*dmui[i]

        Npmu[j, i] = Npmu[j - 1, i] + n_g * ds[j] / mu_local_plus[j, i] if j > 0 else n_g * ds[j] / mu_local_plus[j, i]

print(Npmu.shape)
print("ds shape", ds.shape)
print("ds shape", dsm.shape)
Nmmu  = np.zeros((size, len(mu_ism)))
dmu_minus = np.zeros((size, len(mu_ism)))
mu_local_minus = np.zeros((size, len(mu_ism)))
bfield = bfield[::-1]
numb_density = numb_density[::-1]
B_ism     = bfield[0]

for i, mui_ism in enumerate(mu_ism):

    for j in range(size):

        n_g = numb_density[j]
        Bsprime = bfield[j]
        mu_local2 = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        
        if (mu_local2 <= 0):
            break
        
        mu_local_minus[j,i] = np.sqrt(mu_local2)

        dmu_minus[j,i] = (mui_ism/mu_local_minus[j,i])*(Bsprime/B_ism)*dmui[i]
        
        Nmmu[j, i] = Nmmu[j - 1, i] + n_g * dsm[j] / mu_local_minus[j, i] if j > 0 else n_g * ds[j] / mu_local_minus[j, i]

""" Ionization Rate for N = N(s) """

print(dmu_plus.shape, dmui.shape)
print("mu_i:", mu_ism.shape)
print("N(mu_i):", Npmu[:,0].shape)
print("mu(mu_i, Nj):", mu_local_plus.shape)
print("J(mu_i, Nj)dmui:", dmu_plus.shape)

zeta_plus_mui = np.zeros_like(Npmu)
zeta_plus = np.zeros_like(distance)
Loss = np.ones_like(Npmu)

for l, mui in enumerate(mu_ism):

    for j, Nj in enumerate(Npmu[:,i]):
        mu_ = mu_local_plus[j,l]
        if mu_ <= 0:
            break

        jl_dE = 0.0

        #  Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) 
        Ei = ((energy)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj/mu_)**(1 / (1 + d))

        isms = ism_spectrum(Ei)                        # log_10(j_i(E_i))
        llei = loss_function(Ei)                       # log_10(L(E_i))

        jl_dE = np.sum(isms*llei*diff_energy)  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
        zeta_plus_mui[j, l] = jl_dE / epsilon

print(dmu_plus.shape,dmui.shape,zeta_plus_mui.shape)

zeta_plus = np.sum(dmu_plus * zeta_plus_mui, axis = 1)

zeta_minus_mui = np.zeros_like(Npmu)
zeta_minus = np.zeros_like(distance)

for l, mui in enumerate(mu_ism):
    # initial mu_i conveys a mu(mu_i, N)

    for j, Nj in enumerate(Nmmu[:,l]):
        
        mu_ = mu_local_minus[j,l]
        if mu_ == 0:
            break

        jl_dE = 0.0

        #  Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) 
        Ei = ((energy)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj/mu_)**(1 / (1 + d))

        isms = ism_spectrum(Ei)                        # log_10(j_i(E_i))
        llei = loss_function(Ei)                       # log_10(L(E_i))
        jl_dE = np.sum(isms*llei*diff_energy)  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
    
        zeta_minus_mui[j, l] = jl_dE / epsilon

zeta_minus = np.sum(dmu_minus * zeta_minus_mui, axis = 1) 

for l, mui in enumerate(mu_ism):
    maskp = Npmu[:,l] != 0
    maskm = Nmmu[:,l] != 0
    
    plt.plot(Npmu[maskp,l], zeta_plus_mui[maskp,l],label=f'$\zeta_+(N, {np.round(mui, 4)}) $')
    plt.plot(Nmmu[maskm,l], zeta_minus_mui[maskm,l],label=f'$\zeta_-(N, {np.round(mui, 4)}) $')

plt.title(r"$\zeta_{\pm}(N, \mu) = \frac{1}{2 \varepsilon} \int_0^\infty j_i(E_i, \mu, N) L(E_i) \, dE$", fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.ylim(1.0e-19,1.0e-14)  # Set y-axis limits
plt.savefig('images/zeta_n_mui.png')
plt.show()

plt.plot(distance, zeta_plus + zeta_minus[::-1],label=f'$\zeta_+(s)+\zeta_-(s) $', alpha = 0.5)
plt.plot(distance, zeta_plus,label=f'$\zeta_+(s)+\zeta_-(s) $', alpha = 0.5)
plt.plot(distance, zeta_minus[::-1],label=f'$\zeta_-(s) $', alpha = 0.5)

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title(r"$\zeta_\pm(N) = \int_0^1 \zeta_{\pm}(N, \mu) \, d(\mu(\mu_i))$", fontsize=12)
#plt.ylim(1.0e-19,1.0e-14)  # Set y-axis limits
plt.savefig('images/zeta_n.png')
plt.show()

zeta_mirr = np.zeros_like(Npmu)

for j, Nj in enumerate(Npmu[:,0]):
    
    jl_dE = 0.0
    
    for k, E in enumerate(energy): 

        Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) # E_i(E, N)

        isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
        llei = loss_function(Ei)           # log_10(L(E_i))
        jl_dE += isms*llei*diff_energy[k]  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
    
    zeta_mirr[j] = jl_dE / epsilon             # jacobian * jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon

if False:
    plt.plot(distance, zeta_plus, label=r"$\zeta_+(s)$")    
    #plt.plot(distance, zeta_minus[::-1], label=r"$\zeta_-(s)$")    
    #plt.plot(distance, zeta_mirr, label=r"$\zeta_{mirr}(s)$")    

    #plt.plot(distance, zeta_plus, label=r"$\zeta(s)$")    

    # Title and axis labels
    plt.title(r"$\zeta(s)$")
    plt.xlabel("Distance (pc)")
    plt.ylabel(r"Total Ionization")

    # Logarithmic scaling
    plt.xscale('log')
    plt.yscale('log')

    # Show legend with labels
    plt.legend()

    # Display the plot
    plt.show()


ds = np.linalg.norm(np.diff(radius_vector[::-1], axis=0), axis=1)    
bfield = bfield[::-1]
numb_density = numb_density[::-1]


Nmmu  = np.zeros((size, len(mu_ism))) # matrix N x 2
Jacobmmu = np.zeros((size, len(mu_ism)))
mu_local_minus = np.zeros((size, len(mu_ism)))
rev_numb_density = numb_density[::-1]
rev_bfield        = bfield[::-1]
B_ism      = bfield[-1]

for i, mui_ism in enumerate(mu_ism):

    for j in range(size):

        n_g = rev_numb_density[j]
        Bsprime = rev_bfield[j]
        mu_local2 = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        
        if (mu_local2 <= 0):
            break

        mu_local_minus[j,i] = np.sqrt(mu_local2)

        Jacobmmu[j,i] = (mu_local_plus[j,i]/mui_ism)*(Bsprime/B_ism)
        
        Nmmu[j,i] = Nmmu[j-i,i] + n_g*ds[j] / mu_local_minus[j,i]

Nmirback = Nmmu.copy()*0.0
B_max = np.max(bfield)
s_max = np.argmax(bfield[::-1])
B_ism      = bfield[-1]

for i, mui_ism in enumerate(mu_ism): # cosina alpha_i
    for s in range(s_max):            # at s
        N = 0.0#Nmirback[s, i]
        for s_prime in range(s_max-s): # get mirrored at s_mirr; all subsequent points s < s_mirr up until s_max
            # sprime is the integration variable.
            if (bfield[s_prime] > B_ism*(1-mui_ism**2)):
                break
            mu_local = np.sqrt(1 - bfield[s_prime]*(1-mui_ism**2)/B_ism )
            s_mir = s + s_prime
            dens  = numb_density[s:s_mir]
            diffs = ds[s:s_mir] 
            if len(diffs) == 0:
                break
            N += np.cumsum(dens*diffs/mu_local)
        Nmirback[s,i] = N

#NBtotal = np.sum(Nmirback, axis=1) + Nminusmu
#NFtotal = np.sum(Nmirfor, axis=1) +  Nplusmu
#Ntotal = NFtotal + NBtotal[::-1]

plt.plot(distance, Nmirfor)    
plt.title("Column density $N_{mirr}(s)$")
plt.xlabel("Distance (pc)")
plt.ylabel(r"$N_+(s) + N_-(s) + N_{mirr}(s)$")
plt.xscale('log')
plt.yscale('log')

# Show legend for the plot
plt.legend()

# Display the plot
plt.show()

Nmirfor = Npmu.copy()*0.0
B_max = np.max(bfield)
s_max = np.argmax(bfield)
B_ism      = bfield[0]

for i, mui_ism in enumerate(mu_ism): # cosina alpha_i
    for s in range(s_max):            # at s
        N = 0.0
        for s_prime in range(s_max-s):
            #if (bfield[s_prime] > B_ism*(1-mui_ism**2)):
            #    break
            mu_local = np.sqrt(1 - bfield[s_prime]*(1-mui_ism**2)/B_ism )
            s_mir = s + s_prime
            dens  = numb_density[s:s_mir]
            diffs = ds[s:s_mir] 
            if len(diffs) == 0:
                break
            N += np.cumsum(dens*diffs/mu_local)
        Nmirfor[s,i] = N

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