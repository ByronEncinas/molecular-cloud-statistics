from library import *
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import os, sys
import glob

""" Make plots use LaTex """

""" Parameters """

size = 10_000

Ei = 1.0e+0
Ef = 1.0e+15

n0 = 150 #cm−3 and 
k  = 0.5 # –0.7

d = 0.82
a = 0.1 # spectral index either 0.1 from Low Model, or \in [0.5, 2.0] according to free streaming analytical solution.

# mean energy lost per ionization event
epsilon = 35.14620437477293

# Luminosity per unit volume for cosmic rays (eV cm^2)
Lstar = 1.4e-14

# Flux constant (eV^-1 cm^-2 s^-1 sr^-1) C*(10e+6)**(0.1)/(Enot+6**2.8)
Jstar = 2.43e+15*(10e+6)**(0.1)/(500e+6**2.8) # Proton in Low Regime (M. Padovani & A. Ivlev 2015) https://iopscience.iop.org/article/10.1088/0004-637X/812/2/135

# Flux constant (eV^-1 cm^-2 s^-1)
Enot = 500e+6
Jstar = 2.4e+15*(1.0e+6)**(0.1)/(Enot**2.8)

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6

logE0, logEf = 6, 9
energy = np.logspace(logE0, logEf, size)
diff_energy = np.array([energy[k]-energy[k-1] for k in range(len(energy))])
diff_energy[0] = energy[0]

#ism_spectrum = lambda x: C*(x**0.1/((Estar + x)**2.8))
#ism_spectrum = lambda x: Jstar*(x/Estar)**a
C, alpha, beta, Enot = select_species('L')

ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))
loss_function = lambda z: Lstar*(Estar/z)**d

Neff = np.logspace(19, 27, size)

if True:
    # only for protons
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
    if True:
        plt.plot(Neff, log_H(Neff), label = 'Model H', linestyle='-', color='black')
        plt.plot(Neff, log_L(Neff), label = 'Model L', linestyle='--', color='black')
        plt.xscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig('./images/i_rate/padovani2018', dpi=300)
        plt.close()

    mu_ism = np.logspace(-2, 0, num=50) #np.linspace(0, 1, mus) # np.array([1.0])# 

INPUT = 'amb'

#plt.rcParams['text.usetex'] = True

print(os.path.exists(f'./series/data1_{INPUT}.pkl'))

if os.path.exists(f'./series/data1_{INPUT}.pkl'):
    df = pd.read_pickle(f'./series/data1_{INPUT}.pkl')
    df.index.name = 'snapshot'
    df.index = df.index.astype(int)
    df = df.sort_index()
    x_input   = df["x_input"].to_numpy()[-1]*pc_to_cm
    directions= df["directions"].to_numpy()[-1]
    fields    = df["B_s"].to_numpy()[-1]
    densities = df["n_s"].to_numpy()[-1]
    vectors   = df["r_s"].to_numpy()[-1]*pc_to_cm


def reduction_at_path(vectors,dens,fields,threshold): # input a profile at a time
    
    r_at_s = []
    x_at_s = []
    y_at_s = []
    for i in range(fields.shape[1]):

        vector = vectors[:,i,:]
        den   = dens[:,i]
        field = fields[:,i]

        mask = np.where(den > threshold)[0]

        if mask.size > 0:
            start, end = mask[0], mask[-1]
            #print(mask10)
            #print(i, start, end)
            vector = vector[start:end+1]
            den   = den[start:end+1]
            field = field[start:end+1]
        
        for p_i, B_r in enumerate(field):

            """  
            R = 1 - \sqrt{1 - B(s)/Bl}
            s = distance traveled inside of the molecular cloud (following field lines)
            Bs= Magnetic Field Strenght at s
            """
            pocket, global_info = pocket_finder(field, den, 0, plot=False)
            index_pocket, field_pocket = pocket[0], pocket[1]

            from collections import Counter
            counter = Counter(field.ravel())  # ravel() flattens the array
            most_common_value, count = counter.most_common(1)[0]

            # are there local maxima around our point? 
            try:
                closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
                B_l = min([field[closest_values[0]], field[closest_values[1]]])
                B_h = max([field[closest_values[0]], field[closest_values[1]]])
                # YES! 
                success = True  
            except:
                # NO :c
                R = 1.
                success = False 

            if success:
                # Ok, our point is between local maxima, is inside a pocket?
                if B_r / B_l < 1:
                    # double YES!
                    R = 1. - np.sqrt(1 - B_r / B_l)
                else:
                    # NO!
                    R = 1.

            r_at_s   += [R]
        x_at_s   += vector[:, 0].tolist() #np.linalg.norm(vector, axis=1).tolist()
        y_at_s   += vector[:, 1].tolist()

        #y_at_s   += [X[i,1]]
        #print(len(r_at_s))

    r_at_s = np.array(r_at_s) 
    x_at_s = np.array(x_at_s) 
    y_at_s = np.array(y_at_s) 

    mask = r_at_s < 1

    r_at_s = r_at_s[mask]
    x_at_s = x_at_s[mask]
    y_at_s = y_at_s[mask]

    fig, ax = plt.subplots()

    ax.scatter(x_at_s, r_at_s, s=5, color="red", label=fr"$R(r)$",alpha=0.1)
    ax.set_title(r"Reduction Factor at r")
    ax.set_xlabel("Distance (*)")
    ax.set_ylabel(r"$R(s)$")
    ax.set_xscale('log')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'./images/reduction/r_at_x.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots()

    ax.scatter(x_at_s, y_at_s, s=5, color="red", label=fr"$R_{{<1}}(x,y)$", alpha=1.0)
    ax.set_title(r"$R<1$ at $Proj(x,y)$ ")
    ax.set_xlabel("x (*)")
    ax.set_ylabel("y (*)")
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'./images/reduction/subunity_at_xy.png', dpi=300)
    plt.close()

    return r_at_s

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
            dmu[j,i] = dmui[i]*(mui_ism/mu_local[j,i])*(Bsprime/B_ism)
            Nmu[j, i] = Nmu[j - 1, i] + n_g * ds[j] / mu_local[j, i] if j > 0 else n_g * ds[j] / mu_local[j, i]

    if False:
        fig, ax = plt.subplots()

        for mui in range(Nmu.shape[1]):
            ax.plot(mu_local[:, mui], label=fr"$\mu_i = {mu_ism[mui]}$")

        ax.set_title(r"Evolution of $cos(\alpha(\alpha_i), s)$ along trajectory")
        ax.set_xlabel("Index (Visualization Graph)")
        ax.set_ylabel(r"$\cos(\alpha(\alpha_i), s)$")
        ax.set_xscale('log')
        #ax.legend()
        fig.tight_layout()

        plt.savefig(f'./images/i_rate/Model{model}mu_local_{direction}.png', dpi=300)
        plt.close()

    return Nmu, mu_local, dmu, trajectory

def mirrored_column_density(radius_vector, magnetic_field, numb_density, Nmu, direction=''): 

    ds    = np.insert(np.linalg.norm(np.diff(radius_vector, axis=0), axis=1), 0, 0.0)
    Nmir  = np.zeros((len(magnetic_field), len(mu_ism)))
    s_max = np.argmax(magnetic_field) 
    if  'fwd' in direction:
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

    if False:
        for i, mu in enumerate(mu_ism):
            plt.plot(Nmir[:, i], label=fr'$\mu_{{\mathrm{{ISM}}}}={mu:.2f}$')
        if direction == 'fwd':
            plt.title("Column density $N_{mirr}^+(s)$")
        if direction == 'bwd':
            plt.title("Column density $N_{mirr}^-(s)$")
        plt.xlabel("Steps")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'./images/i_rate/model{model}N{direction}.png', dpi=300)
        plt.close()

    return Nmir

def ionization_rate(Nmu, mu_local, dmu, direction = ''):
    zeta_mui = np.zeros_like(Nmu)
    zeta = np.zeros_like(Nmu[:,0])
    jspectra = np.zeros((Nmu.shape[0], energy.shape[0]))

    for l, mui in enumerate(mu_ism):

        for j, Nj in enumerate(Nmu[:,l]):
            mu_ = mu_local[j,l]
            if mu_ <= 0:
                break

            #  Ei = ((E)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj)**(1 / (1 + d)) 
            Ei = ((energy)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj/mu_)**(1 / (1 + d))

            isms = ism_spectrum(Ei)                        # log_10(j_i(E_i))
            llei = loss_function(Ei)                       # log_10(L(E_i))
            sigma_ion = cross(energy)
            spectra   = 0.5*isms*llei/loss_function(energy)  
            #jspectra[j,:] = spectra
            #jl_dE = np.sum(isms*llei*diff_energy)  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
            #zeta_mui[j, l] = jl_dE / epsilon 
            zeta_mui[j, l] = np.sum(spectra*sigma_ion*diff_energy)

    zeta = np.sum(dmu * zeta_mui, axis = 1)

    if False:
        if 'fwd' in direction:
            plt.plot(zeta_mui, label=f'$\zeta_+(s) $', alpha = 0.5)
        if 'bwd' in direction:
            plt.plot(zeta_mui, label=f'$\zeta_-(s) $', alpha = 0.5)
        plt.title(r"$\zeta(N, \mu)$", fontsize=12)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'./images/i_rate/Model{model}zeta_{direction}.png', dpi=300)
        plt.close()

    return zeta, zeta_mui, jspectra


if __name__ == '__main__':
    lines = x_input.shape[0]

    zeta_at_x = np.zeros(lines)
    nmir_at_x = np.zeros(lines)

    local_spectra_at_x = np.zeros((lines, size))

    zeta_full = []
    nmir_full = []
    fig, ax = plt.subplots()


    for line in range(lines):
        density    = densities[:, line]
        field =  fields[:, line]*1e6
        vector  =  vectors[:, line, :]

        # slice out zeroes        
        mask = np.where(density > 0.0)[0]
        start, end = mask[0], mask[-1]

        density    = density[start:end]
        field =  field[start:end] #np.ones_like(field[start:end]) 
        vector  =  vector[start:end, :]
        trajectory = np.cumsum(np.linalg.norm(vector, axis=1))*pc_to_cm #np.insert(, 0, 0.0)
        
        try:
            xi_input  = x_input[line]
            arg_input = np.where(xi_input[0] == vector[:,0])[0][0]
        except:
            raise IndexError("[Error] arg_input was removed during slicing")
        
        """ Column Densities N_+(mu, s) & N_-(mu, s)"""

        Npmu, mu_local_fwd, dmu_fwd, t_fwd = column_density(vector, field, density, "fwd")
        Nmmu, mu_local_bwd, dmu_bwd, t_bwd = column_density(vector[::-1, :], field[::-1], density[::-1], "bwd")

        Nmir_fwd = mirrored_column_density(vector, field, density, Npmu, 'mir_fwd')
        Nmir_bwd = mirrored_column_density(vector[::-1,:], field[::-1], density[::-1], Nmmu, 'mir_bwd')

        """ Ionization Rate for N = N(s) """
        
        zeta_mir_fwd, zeta_mui_mir_fwd, spectra_fwd  = ionization_rate(Nmir_fwd, mu_local_fwd, dmu_fwd, 'mir_fwd')
        zeta_mir_bwd, zeta_mui_mir_bwd, spectra_bwd  = ionization_rate(Nmir_bwd, mu_local_bwd, dmu_bwd, 'mir_bwd')

        Nmir = np.sum(Nmir_fwd + Nmir_bwd[::-1], axis=1) # not that relevant

        zeta = (zeta_mir_fwd+ zeta_mir_bwd[::-1])
        local_spec = np.sum((spectra_fwd+ spectra_bwd[::-1]), axis=0) # Adding the corresponding 

        zeta_at_x[line] = zeta[arg_input]
        nmir_at_x[line] = Nmir[arg_input]
        local_spectra_at_x[line, :] = local_spec
        ax.scatter(nmir_at_x[line], zeta_at_x[line], marker='|', color="r")

    ax.plot(Neff, log_L(Neff), "-", color="black", label=fr"$\zeta_L$ Fit")
    ax.plot(Neff, log_H(Neff), "-", color="black", label=fr"$\zeta_H$ Fit")

    ax.set_xlabel("Distance (cm)", fontsize=16)
    ax.set_ylabel("$\log_{10}(\zeta / s^{-1})$", fontsize=16)
    ax.set_xscale('log')

    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'./Model{model}_marginal.png', dpi=300)
    plt.close(fig)

    exit()
    from scipy.stats import binned_statistic_2d
    from matplotlib.colors import LogNorm

    if True:
        x = x_input[:line+1,0]
        y = x_input[:line+1,1]
        z = x_input[:line+1,2]

        if x.shape[0] < 100:
            bin = 25
        else:
            bin = np.ceil(np.sqrt(x.shape[0]))

        stat, x_edges, y_edges, _ = binned_statistic_2d(x, y, zeta_at_x, statistic='mean', bins=bin)

        plt.figure(figsize=(6, 6))
        im = plt.imshow(stat.T, origin='lower',
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        aspect='equal', cmap='plasma',
                        norm=LogNorm(vmin=np.nanmin(stat[stat > 0]),  # avoid log(0)
                                    vmax=np.nanmax(stat)))

        plt.colorbar(im, label=r"$\zeta $")
        plt.scatter(x, y, s=1, c='white', alpha=0.05)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Projection of Scatter Data (Log Colorbar)")
        plt.savefig(f'./images/i_rate/Model{model}_ionization_rate_map.png', dpi=300)

    if False:
        fig, ax = plt.subplots()

        ax.scatter(nmir_full, zeta_full, s=8, marker='|', color="red", label=fr"$\zeta$")
        #ax.plot(Nmir,label=fr"$\zeta$")
        ax.plot(Neff, 10**log_H(Neff), label='Model H')
        ax.plot(Neff, 10**log_L(Neff), label='Model L')
        ax.set_title(r"Total Ionization at X")
        ax.set_xlabel("Distance (cm)")
        ax.set_ylabel(r"$\zeta(s)$")
        ax.set_yscale('log')
        ax.set_xscale('log')
        #ax.legend()
        fig.tight_layout()
        plt.grid(True)
        plt.savefig(f'./images/i_rate/Model{model}_ionization_rate0.png', dpi=300)
        plt.close()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    print("zeta_at_x.shape = ", zeta_at_x.shape)

    ax.hist(np.log10(zeta_at_x), bins=zeta_at_x.shape[0]//10, alpha=1,
            histtype='stepfilled', label=f"{time} Myrs")

    print("Mean power of $\log_{10}(\zeta}$: ", np.mean(np.log10((zeta_at_x))))

    ax.set_yscale('log')
    ax.set_ylabel('Counts', fontsize=16)
    ax.set_xlabel('''$\log_{10}(\zeta / s^{-1})$''', fontsize=12)
    ax.set_xlim(np.max(np.log10(zeta_at_x)), np.max(np.log10(zeta_at_x)))
    ax.legend(frameon=False)
    ax.set_title('''Histogram $\log_{10}(\zeta / s^{-1})$''')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./images/i_rate/Model{model}_histogram.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()

    #ax.scatter(nmir_reduced, zeta_reduced, s=8, marker='x', color="red", label=fr"$\zeta$")
    ax.scatter(nmir_at_x, np.log10(zeta_at_x), s=5, marker='x', color="red", label=fr"$\zeta$", alpha=0.2)
    ax.plot(Neff, log_H(Neff), label='Model H', linestyle='-', color='black')
    ax.plot(Neff, log_L(Neff), label='Model L', linestyle='--', color='black')
    ax.set_title("Total Ionization at X", fontsize=16)
    ax.set_xlabel("Distance (cm)", fontsize=16)
    ax.set_ylabel("$\zeta(s)$", fontsize=16)
    ax.set_xscale('log')
    ax.set_ylim(-23.5, -15.5)
    #ax.legend()
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'./images/i_rate/Model{model}_ionization_rate1.png', dpi=300)
    plt.close(fig)

