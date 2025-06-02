from library import *
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import glob

""" Parameters """

pc_to_cm = 3.086 * 10e+18  # cm per parsec

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
for i,Ni in enumerate(Neff):
    lzl = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_L)] )
    logzl.append(lzl)
#print("Extremes of fitting zeta(N) ", logzl[0], logzl[-1])

#logzetalfit = np.array(logzl)

logzh = []

for i,Ni in enumerate(Neff):
    lzh = sum( [cj*(np.log10(Ni))**j for j, cj in enumerate(model_H)] )
    logzh.append(lzh)

#print("Extremes of fitting zeta(N) ", logzh[0], logzh[-1])

#logzetahfit = np.array(logzh)

#svnteen = np.ones_like(Neff)*(-17)

from scipy import interpolate

log_L = interpolate.interp1d(Neff, logzl)
log_H = interpolate.interp1d(Neff, logzh)
plt.plot(Neff, log_H(Neff), label = 'Model H')
plt.plot(Neff, log_L(Neff), label = 'Model L')
#plt.plot(Neff, logzh, label = 'Model H')
#plt.plot(Neff, logzl, label = 'Model L')
plt.xscale('log')
plt.legend()
plt.savefig('./images/padovani2018')
plt.close()

if False:
    print("N_path/N_los = 1/10: ", 0.1)
    zeta_los = 10**log_L(1.0e+23)
    zeta_path = 10**log_L(1.0e+22)
    print(zeta_los)
    print(zeta_path)
    print(zeta_path/zeta_los)

    print("N_path/N_los = 1/10: ", 0.1)
    zeta_los = 10**log_H(1.0e+23)
    zeta_path = 10**log_H(1.0e+22)
    print(zeta_los)
    print(zeta_path)
    print(zeta_path/zeta_los)

if True:
    origin_folder = 'thesis_stats/amb/300/DataBundle119231.npz'

    line = int(sys.argv[-1])

    data = np.load(origin_folder, mmap_mode='r')
    radius_vector = data['positions'][:,line,:]
    trajectory = data['trajectory'][:,line]
    numb_density = data['number_densities'][:,line]
    magnetic_field = data['magnetic_fields'][:,line]
    mask = numb_density > 0
    
    radius_vector = radius_vector[mask]
    trajectory = trajectory[mask]
    numb_density = numb_density[mask]
    magnetic_field = magnetic_field[mask]

    import numpy as np, matplotlib.pyplot as plt

    fig, axs = plt.subplot_mosaic([["a"], ["b"], ["c"]], figsize=(10, 8), layout="constrained")
    axs["a"].plot(trajectory, c="b"); axs["a"].set_title("Trajectory"); axs["a"].set_ylabel("Trajectory")
    axs["b"].plot(numb_density, c="g"); axs["b"].set_title("Number Density"); axs["b"].set_ylabel("n (cm$^{-3}$)")
    axs["c"].plot(magnetic_field, c="r"); axs["c"].set_title("Magnetic Field"); axs["c"].set_ylabel("B (μG)"); axs["c"].set_xlabel("Index")
    plt.savefig("./images/mosaic_plot.png"); plt.close()



""" Column Densities N_+(mu, s) & N_-(mu, s)"""

mu_ism = np.logspace(-1, 0, num=5) #np.array([1.0])#
dmui = np.insert(np.diff(mu_ism), 0, mu_ism[0])
ds  = np.insert(np.diff(trajectory), 0, 0.0)
dsm = np.insert(np.linalg.norm(np.diff(radius_vector[::-1], axis=0), axis=1), 0, 0.0)

Npmu  = np.zeros((len(magnetic_field), len(mu_ism)))
dmu_plus = np.zeros((len(magnetic_field), len(mu_ism)))
mu_local_plus = np.zeros((len(magnetic_field), len(mu_ism)))
B_ism     = magnetic_field[0]


for i, mui_ism in enumerate(mu_ism):

    for j in range(len(magnetic_field)):

        n_g = numb_density[j]
        Bsprime = magnetic_field[j]
        mu_local2 = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        
        if (mu_local2 <= 0):
            break
        
        mu_local_plus[j,i] = np.sqrt(mu_local2)
        dmu_plus[j,i] = (mui_ism/mu_local_plus[j,i])*(Bsprime/B_ism)*dmui[i]


        Npmu[j, i] = Npmu[j - 1, i] + n_g * ds[j] / mu_local_plus[j, i] if j > 0 else n_g * ds[j] / mu_local_plus[j, i]

print(Npmu.shape)
print("ds shape", ds.shape)
print("ds shape", dsm.shape)
Nmmu  = np.zeros((len(magnetic_field), len(mu_ism)))
dmu_minus = np.zeros((len(magnetic_field), len(mu_ism)))
mu_local_minus = np.zeros((len(magnetic_field), len(mu_ism)))
magnetic_field = magnetic_field[::-1]
numb_density = numb_density[::-1]
B_ism     = magnetic_field[0]

for i, mui_ism in enumerate(mu_ism):

    for j in range(len(magnetic_field)):

        n_g = numb_density[j]
        Bsprime = magnetic_field[j]
        mu_local2 = 1 - (Bsprime/B_ism) * (1 - mui_ism**2)
        
        if (mu_local2 <= 0):
            break
        
        mu_local_minus[j,i] = np.sqrt(mu_local2)

        dmu_minus[j,i] = (mui_ism/mu_local_minus[j,i])*(Bsprime/B_ism)*dmui[i]
        
        Nmmu[j, i] = Nmmu[j - 1, i] + n_g * dsm[j] / mu_local_minus[j, i] if j > 0 else n_g * ds[j] / mu_local_minus[j, i]


if True:
    for mui in range(Npmu.shape[1]):
        plt.plot(trajectory, mu_local_plus[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
        plt.legend()

    plt.title(r"Evolution of $cos(\alpha(\alpha_i), s)$ along trajectory")
    plt.xlabel("trajectory (pc)")
    plt.ylabel(r"$cos(\alpha(\alpha_i), s)$")
    plt.xscale('log')
    plt.savefig('images/mu_local_plus.png')  # Save as PNG with high resolution
    plt.close()
    for mui in range(Npmu.shape[1]):
        plt.plot(trajectory, Npmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}")    
        plt.legend()

    plt.title("Column density $N_+(\mu_i)$  ")
    plt.xlabel("trajectory (pc)")
    plt.ylabel(r"$N_+(\mu_i, s)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('images/Nplus.png')  # Save as PNG with high resolution
    plt.close()

    for mui in range(Npmu.shape[1]):
        plt.plot(trajectory, Nmmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}")    
        plt.legend()

    plt.title("Column density $N_-(\mu_i)$  ")
    plt.xlabel("trajectory (pc)")
    plt.ylabel(r"$N_-(\mu_i, s)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('images/Nminus.png')  # Save as PNG with high resolution
    plt.close()
    for mui in range(dmu_plus.shape[1]):
        plt.plot(trajectory, dmu_plus[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
        plt.legend()

    plt.title(r"Evolution of $\frac{B}{B_i}\frac{\mu_i}{\mu}$ (Forwards)")
    plt.xlabel("trajectory (cm)")
    plt.ylabel(r"$\frac{\partial \mu_i}{\partial \mu}$")
    plt.xscale('log')
    plt.savefig('images/Jplus.png')  # Save as PNG with high resolution
    plt.close()
    
    for mui in range(dmu_minus.shape[1]):
        plt.plot(trajectory, dmu_minus[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
        plt.legend()

    plt.title(r"Evolution of $\frac{B}{B_i}\frac{\mu_i}{\mu}$ (Backwards)")
    plt.xlabel("trajectory (cm)")
    plt.ylabel(r"$\frac{\partial \mu_i}{\partial \mu}$")
    plt.xscale('log')
    plt.savefig('images/Jminus.png')  # Save as PNG with high resolution
    plt.close()

""" Ionization Rate for N = N(s) """

print(dmu_plus.shape, dmui.shape)
print("mu_i:", mu_ism.shape)
print("N(mu_i):", Npmu[:,0].shape)
print("mu(mu_i, Nj):", mu_local_plus.shape)
print("J(mu_i, Nj)dmui:", dmu_plus.shape)

zeta_plus_mui = np.zeros_like(Npmu)
zeta_plus = np.zeros_like(trajectory)
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
zeta_minus = np.zeros_like(trajectory)

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

Neff = np.logspace(19, 22.5, size)

plt.plot(Neff, 10**log_L(Neff), label = 'Model L')
plt.plot(Neff, 10**log_H(Neff), label = 'Model H')


for l, mui in enumerate(mu_ism):
    maskp = Npmu[:,l] != 0
    maskm = Nmmu[:,l] != 0
    
    plt.plot(Npmu[maskp,l], zeta_plus_mui[maskp,l]) # ,label=f'$\zeta_+(N, {np.round(mui, 4)}) $'
    #plt.plot(Nmmu[maskm,l], zeta_minus_mui[maskm,l],label=f'$\zeta_-(N, {np.round(mui, 4)}) $')

plt.title(r"$\zeta_{\pm}(N, \mu) = \frac{1}{2 \varepsilon} \int_0^\infty j_i(E_i, \mu, N) L(E_i) \, dE$", fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()
#plt.ylim(1.0e-19,1.0e-14)  # Set y-axis limits
plt.savefig('images/zeta_n_mui.png')
plt.close()

plt.plot(trajectory, zeta_plus + zeta_minus[::-1],label=f'$\zeta_+(s)+\zeta_-(s) $', alpha = 0.5)

plt.xscale('log')
plt.yscale('log')
plt.legend()

#plt.ylim(1.0e-19,1.0e-14)  # Set y-axis limits
plt.savefig('images/zeta_n.png')
plt.close()

zeta_mirr = np.zeros_like(Npmu)

for j, Nj in enumerate(Npmu[:,0]):
    
    jl_dE = 0.0
    
    for k, E in enumerate(energy): 

        Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) # E_i(E, N)

        isms = ism_spectrum(Ei)            # log_10(j_i(E_i))
        llei = loss_function(Ei)           # log_10(L(E_i))
        jl_dE += isms*llei*diff_energy[k]  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
    
    zeta_mirr[j] = jl_dE / epsilon             # jacobian * jl_dE/epsilon #  \sum_k j_i(E_i)L(E_i) \Delta E_k / \epsilon

Nmir_plus = Npmu.copy()*0.0
B_max = np.max(magnetic_field)
s_max = np.argmax(magnetic_field)
B_ism      = magnetic_field[0]

for i, mui_ism in enumerate(mu_ism): # cosina alpha_i
    for s in range(s_max):            # at s
        N = Npmu[s, i]
        for s_prime in range(s_max-s): # get mirrored at s_mirr; all subsequent points s < s_mirr up until s_max
            # sprime is the integration variable.
            if (magnetic_field[s_prime] > B_ism*(1-mui_ism**2)):
                break
            mu_local = np.sqrt(1 - magnetic_field[s_prime]*(1-mui_ism**2)/B_ism )
            s_mir = s + s_prime
            dens  = numb_density[s:s_mir]
            diffs = ds[s:s_mir] 
            N += np.cumsum(dens*diffs/mu_local)
        Nmir_plus[s,i] = N

plt.plot(Nmir_plus)    
plt.title("Column density $N_{mirr}(s)$")
plt.xlabel("Steps")
plt.yscale('log')
plt.legend()
plt.savefig('./images/Nmirr_f.png')
plt.close()

magnetic_field = magnetic_field[::-1]
numb_density = numb_density[::-1]
Nmir_minus = Nmmu.copy()*0.0
s_max = np.argmax(magnetic_field)
B_ism      = magnetic_field[0]

for i, mui_ism in enumerate(mu_ism): # cosina alpha_i
    for s in range(s_max):            # at s
        N = Nmmu[s, i]
        for s_prime in range(s_max-s): # get mirrored at s_mirr; all subsequent points s < s_mirr up until s_max
            # sprime is the integration variable.
            if (magnetic_field[s_prime] > B_ism*(1-mui_ism**2)):
                break
            mu_local = np.sqrt(1 - magnetic_field[s_prime]*(1-mui_ism**2)/B_ism )
            s_mir = s + s_prime
            dens  = numb_density[s:s_mir]
            diffs = dsm[s:s_mir] 
            N += np.cumsum(dens*diffs/mu_local)
        Nmir_minus[s,i] = N
        
plt.plot(Nmir_minus)    
plt.title("Column density $N_{mirr}(s)$")
plt.xlabel("Steps")
plt.yscale('log')
plt.legend()
plt.savefig('./images/Nmirr_b.png')
plt.close()

Nmirfor = Npmu.copy()*0.0
B_max = np.max(magnetic_field)
s_max = np.argmax(magnetic_field)
B_ism      = magnetic_field[0]

for i, mui_ism in enumerate(mu_ism): # cosina alpha_i
    for s in range(s_max):            # at s
        N = 0.0
        for s_prime in range(s_max-s):
            #if (magnetic_field[s_prime] > B_ism*(1-mui_ism**2)):
            #    break
            mu_local = np.sqrt(1 - magnetic_field[s_prime]*(1-mui_ism**2)/B_ism )
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
        
            if 1 - (magnetic_field[j]/magnetic_field[0])*(1 - mui**2) > 0:
                mu_local = np.sqrt(1 - (magnetic_field[j]/magnetic_field[0])*(1 - mui**2))
                Jacobian = (mui/mu_local)*(magnetic_field[j]/magnetic_field[0])    
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
plt.plot(trajectory[not_nan],log_forward_ion_rate[-1, not_nan], label=r'$\log(y)$ vs $x$', color='b')

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
plt.close()

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
plt.close()