from library import *
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import os, sys
import glob

""" Make plots use LaTex """

#plt.rcParams['text.usetex'] = True

model, num_file = sys.argv[1], sys.argv[2]

case = 'ideal'

print(sys.argv)
print(model, num_file)

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
if model == 'L':
    C = 2.43e+15 *4*np.pi
    alpha, beta = 0.1, 2.8
    Enot = 650e+6
elif model == 'H':
    C = 2.43e+15 *4*np.pi
    alpha, beta = -0.8, 1.9
    Enot = 650e+6
#elif model == 'e':
#    C = 2.1e+18#4*np.pi
#    alpha, beta = -1.5, 1.7
#    Enot = 710e+6

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

if True:

    import h5py
    import numpy as np

    h5_path = f"./stats/{case}/{num_file}/DataBundle.h5"

    with h5py.File(h5_path, "r") as f:

        # --- Datasets ---
        x_input         = f["starting_point"][()]
        n_rs            = f["number_densities"][()]
        B_rs            = f["magnetic_fields"][()]
        c_rs            = f["column_path"][()]
        reduction_fac   = f["reduction_factor"][()]
        average_column  = f["column_los"][()]

        # --- Metadata / attributes ---
        cores_used     = f.attrs["cores_used"]
        rloc           = f.attrs["rloc"]
        center         = f.attrs["center"]
        volume_range   = f.attrs["volume_range"]
        density_range  = f.attrs["density_range"]
        follow_index0  = f.attrs["index"]
        time           = f.attrs["time"]

        # --- Physical Parameters ---
        numb_densities    = f["densities"][()][:, :]
        magnetic_fields =  f["bfields"][()][:, :]
        radius_vectors  =  f["vectors"][()][:, :, :]*pc_to_cm

    print("Total number of lines: ", magnetic_fields.shape)

    #numb_density    = numb_density[:am]
    #magnetic_field = magnetic_field[:am]
    #radius_vector  =  radius_vector[:am, :]
    #magnetic_field =  np.mean(magnetic_field)*magnetic_field/magnetic_field
    mu_ism = np.logspace(-2, 0, num=50) #np.linspace(0, 1, mus) # np.array([1.0])# 


def re_do_stats_py_plots():
    r_u, r_l = reduction_fac[0,:], reduction_fac[1,:]
    distance = np.linalg.norm(x_input, axis=1)*pc_to_cm


    # size of arrays
    m = magnetic_fields.shape[1]
    N = magnetic_fields.shape[0]

    # reduction factor
    r_u      = np.array(r_u) # threshold * 1
    r_l      = np.array(r_l) # threshold * 10
    subunity = r_u < 1
    unity    = r_u == 1
    rho_subunity = x_input[subunity,:]
    rho_unity = x_input[unity,:]

    # removing PATH/LOS passing through core
    order    = np.argsort(distance)
    N_path_ordered = c_rs[order][1:]
    N_los_ordered = average_column[order][1:]
    s_ordered = distance[order][1:]
    N_path = c_rs.copy()
    N_los  = average_column.copy()
    s      = distance.copy()

    if True: # N_path & N_los Linear fit vs. Distance away from core
        fig, ax = plt.subplots()

        log_y = np.log10(N_los_ordered); m1, b1 = np.polyfit(s_ordered, log_y, 1); fit1 = 10**(m1 * s_ordered + b1)
        log_y2 = np.log10(N_path_ordered); m2, b2 = np.polyfit(s_ordered, log_y2, 1); fit2 = 10**(m2 * s_ordered+ b2)

        eq1 = rf"$\log_{{10}}(N_{{los}}) = {m1:.4e}\,s + {b1:.4f}$"
        eq2 = rf"$\log_{{10}}(N_{{path}}) = {m2:.4e}\,s + {b2:.4f}$"

        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
        ax.set_ylabel(r'$N$ (cm$^{-2}$)', fontsize=16)
        ax.set_yscale('log')
        #ax.set_xscale('log')

        ax.scatter(s_ordered, N_los_ordered, marker='o',color="#8E2BAF", s=5, label=r'$N_{\rm los}$')
        ax.scatter(s_ordered, N_path_ordered, marker ='v',color="#148A02", s=5, label=r'$N_{\rm path}$')
        ax.plot(s_ordered, fit1, '-' , color="black", linewidth=1,label='$N_{los} fit$')
        ax.plot(s_ordered, fit2, '--', color="black", linewidth=1,label='$N_{path} fit$')

        ax.text(0.05, 0.65, eq1, transform=ax.transAxes, color="#8E2BAF", fontsize=12, va="top")
        ax.text(0.05, 0.60, eq2, transform=ax.transAxes, color="#148A02", fontsize=12, va="top")

        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc="upper left", fontsize=12)
        
        plt.title(r"Ratio Column Density (path/los)", fontsize=16)
        fig.tight_layout()
        plt.savefig('./images/columns/column_comparison.png', dpi=300)
        plt.close()

    if True: # Ratio N_path/N_los
        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
        ax.set_ylabel(r'$ratio$', fontsize=16)
        ax.set_yscale('log')
        #ax.set_xscale('log')
        
        ax.scatter(s, N_path/N_los, marker ='v',color="black", s=5, label=r'$N_{\rm path}/N_{\rm los}$', alpha=0.3)
        ax.legend(loc="upper left", fontsize=12)    
        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)
        
        plt.title(r"Ratio $N_{path}/N_{los}$", fontsize=16)
        fig.tight_layout()
        plt.savefig('./images/columns/column_ratio.png', dpi=300)
        plt.close()

    if True: # N_path & N_los vs. Density
        log_x = np.log10(n_rs)
        log_y = np.log10(N_los); m1, b1 = np.polyfit(log_x, log_y, 1); fit1 = 10**(m1 * log_x + b1)
        log_y2 = np.log10(N_path); m2, b2 = np.polyfit(log_x, log_y2, 1); fit2 = 10**(m2 * log_x+ b2)
        
        fig, ax = plt.subplots()

        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax.set_ylabel(r'$N$ (cm$^{-2}$)', fontsize=16)
        ax.set_yscale('log')
        ax.set_xscale('log')

        eq1 = rf"$\log_{{10}}(N_{{los}}) = {m1:.5f}\,\log_{{10}}(n_g) + {b1:.5f}$"
        eq2 = rf"$\log_{{10}}(N_{{path}}) = {m2:.5f}\,\log_{{10}}(n_g) + {b2:.5f}$"
        ax.text(0.05, 0.85, eq1, transform=ax.transAxes, color="#8E2BAF", fontsize=12, va="top")
        ax.text(0.05, 0.80, eq2, transform=ax.transAxes, color="#148A02", fontsize=12, va="top")

        ax.plot(n_rs, fit1, '-', color="black", linewidth=1,label='$N_{los}$')
        ax.plot(n_rs, fit2, '--', color="black", linewidth=1,label='$N_{path}$')

        ax.scatter(n_rs, N_los, marker='o',color="#8E2BAF", s=5, label=r'$N_{\rm los}$', alpha=0.3)
        ax.scatter(n_rs, N_path, marker ='v',color="#148A02", s=5, label=r'$N_{\rm path}$', alpha=0.3)
        
        ax.tick_params(axis='both')
        ax.grid(True, which='both', alpha=0.3)

        plt.title(r"$N(n_g)$", fontsize=16)
        ax.legend(loc="upper right", fontsize=12)
        
        fig.tight_layout()
        plt.savefig('./images/columns/column_density.png', dpi=300)
        plt.close()

    log_ionization_path_l, log_ionization_path_h = ionization_rate_fit(c_rs)
    log_ionization_los_l, log_ionization_los_h  = ionization_rate_fit(average_column)

    if True:

        fig, (ax_l, ax_h) = plt.subplots(1, 2, figsize=(10, 5),gridspec_kw={'wspace': 0, 'hspace': 0}, sharey=True)

        ax_l.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax_l.set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
        ax_l.set_xscale('log')
        ax_l.scatter(n_rs, log_ionization_los_l, marker='o', color="#8E2BAF", s=5, alpha=0.3, label=r'$\zeta_{\rm los}$')
        ax_l.scatter(n_rs, log_ionization_path_l, marker='v', color="#148A02", s=5, alpha=0.3, label=r'$\zeta_{\rm path}$')
        ax_l.grid(True, which='both', alpha=0.3)
        ax_l.legend(fontsize=16)
        ax_l.set_ylim(-19.5, -15.5)
        ax_l.set_title("Model $\mathcal{L}$", fontsize=16)

        ax_h.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax_h.set_xscale('log')
        ax_h.scatter(n_rs, log_ionization_los_h, marker='o', color="#8E2BAF", s=5, alpha=0.3, label=r'$\zeta_{\rm los}$')
        ax_h.scatter(n_rs, log_ionization_path_h, marker='v', color="#148A02", s=5, alpha=0.3, label=r'$\zeta_{\rm path}$')
        ax_h.grid(True, which='both', alpha=0.3)
        ax_h.legend(fontsize=16)
        ax_h.set_ylim(-19.5, -15.5)
        ax_h.set_title("Model $\mathcal{H}$", fontsize=16)
        ax_h.tick_params(labelleft=False)

        fig.suptitle(r"Ionization Rate ($\zeta$) vs. Density ($n_g$)", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('./images/columns/zeta_density_combined.png', dpi=300)
        plt.close()

    if True: # Ionization rate (L) vs. Density

        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)    
        ax.set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
        
        #ax.set_yscale('log')
        ax.set_xscale('log')
        
        ax.scatter(n_rs, log_ionization_los_l, marker='o',color="#8E2BAF", s=5, alpha = 0.3, label=r'$\zeta_{\rm los}$')
        ax.scatter(n_rs, log_ionization_path_l, marker ='v',color="#148A02", s=5, alpha = 0.3, label=r'$\zeta_{\rm path}$')

        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=16)
        ax.set_ylim(-19.5, -15.5)

        fig.tight_layout()
        plt.savefig(f'./images/columns/zeta_density_l.png', dpi=300)
        plt.close()

    if True: # Ionization rate (H) vs. Density

        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax.set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
        
        ax.set_xscale('log')
        
        ax.scatter(n_rs, log_ionization_los_h, marker='o',color="#8E2BAF", s=5, alpha = 0.3, label=r'$\zeta_{\rm los}$')
        ax.scatter(n_rs, log_ionization_path_h, marker ='v',color="#148A02", s=5, alpha = 0.3, label=r'$\zeta_{\rm path}$')

        ax.grid(True, which='both', alpha=0.3)
        plt.title(rf"$\zeta(n_g)$ ({case + num_file})", fontsize=16)
        ax.legend(fontsize=16)
        ax.set_ylim(-19.5, -15.5)
        fig.tight_layout()
        plt.savefig(f'./images/columns/zeta_density_h.png', dpi=300)
        plt.close()

    ratio_ionization_path_to_los_l = log_ionization_path_l-log_ionization_los_l
    ratio_ionization_path_to_los_h = log_ionization_path_l-log_ionization_los_h

    if True: # Ratio Ionization_path/Ionization_los vs. Density
        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
        ax.set_ylabel(r'Ratio', fontsize=16)
        ax.set_xscale('log')
        
        ax.scatter(n_rs, ratio_ionization_path_to_los_l, marker='o',color="#CB71EA", s=5, alpha = 0.3, label='Model $\mathcal{L}$')
        ax.scatter(n_rs, ratio_ionization_path_to_los_h, marker ='v',color="#60D24F", s=5, alpha = 0.3, label='Model $\mathcal{H}$')

        ax.grid(True, which='both', alpha=0.3)
        
        plt.title(rf"$\zeta_{{path}}/\zeta_{{los}} $ ({case + num_file})", fontsize=16)
        ax.set_ylim(-5, 5)

        ax.legend(fontsize=16)
        fig.tight_layout()
        plt.savefig(f'./images/columns/ratio_zeta_density.png', dpi=300)
        plt.close()

    if True: # Ratio Ionization_path & Ionization_los vs. Distance
        fig, ax = plt.subplots()
        
        ax.scatter(distance, ratio_ionization_path_to_los_l, marker='o', color="#CB71EA", s=5, alpha = 0.5, label='Model $\mathcal{L}$')
        ax.scatter(distance, ratio_ionization_path_to_los_h, marker ='v', color="#60D24F", s=5, alpha = 0.5, label='Model $\mathcal{H}$')

        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)    
        ax.set_ylabel(r'Ratio', fontsize=16)
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim(-5, 5)
        plt.title(rf"$\zeta_{{path}}/\zeta_{{los}} $ ({case + num_file})", fontsize=16)
        ax.legend(fontsize=12)
        fig.tight_layout()
        plt.savefig('./images/columns/ratio_zeta_distance.png', dpi=300)
        plt.close()

    if True: # Reduction factor vs. R -- (R < 1) Scatter and Histogram
        s_subunity = distance[subunity]
        r_u_subunity = r_u[subunity]
        
        fig, ax = plt.subplots()
        
        ax.set_xlabel(r'''Distance $r$ (cm)''', fontsize=16)
        ax.set_ylabel(r'$1-R$', fontsize=16)
        #ax.set_yscale('log')
        ax.set_xscale('log')

        ax.scatter(s_subunity, r_u_subunity, marker ='o',color='black', s=5, label="$R<1$")

        ax.grid(True, which='both', alpha=0.3)
        
        plt.title(rf"$1 - R(s)$  ({case + num_file})", fontsize=16)
        ax.legend(fontsize=12)
        
        fig.tight_layout()
        plt.savefig('./images/reduction/r_subunity_complement.png', dpi=300)
        plt.close()

        # 1 - R Histogram vs. distance
        print("r_subunity.shape = ", r_u_subunity.shape)
        fig, ax = plt.subplots(figsize=(8, 3.5))

        bins = np.linspace(
            np.min(1 - r_u_subunity),
            np.max(1 - r_u_subunity),
            r_u_subunity.shape[0] // 10
        )

        ax.hist(1 - r_u_subunity, bins=bins, alpha=1,
                histtype='stepfilled', label=f"{time} Myrs")

        print("Mean $R<1$: ", np.mean((1 - r_u_subunity)))
        plt.title(rf"$1 - R(s)$  ({case + num_file})", fontsize=16)
        ax.set_yscale('log')
        ax.set_ylabel('Counts', fontsize=16)
        ax.set_xlabel('''$\log_{10}(\zeta / s^{-1})$''', fontsize=12)
        ax.legend(fontsize=12)

        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./images/reduction/r_complement_histogram.png', dpi=300)


    if x_input.shape[0] > 100: # Note to self, for small rloc < 1 this code doesnt take long
        # Subunity
        fig, ax = plt.subplots()
        ax.scatter(rho_subunity[:, 0], rho_subunity[:, 1], color='red', s=8, alpha=0.3, label="$R<1$")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig('./images/xyz_distro/xy_subunity.png', dpi=300)
        plt.close(fig)

        # Unity
        fig, ax = plt.subplots()
        ax.scatter(rho_unity[:, 0], rho_unity[:, 1], color='red', s=8, alpha=0.3, label="$R=1$")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig('./images/xyz_distro/xy_unity.png', dpi=300)
        plt.close(fig)

        # Both
        fig, ax = plt.subplots()
        ax.scatter(rho_unity[:, 0], rho_unity[:, 1], color='red', s=8, alpha=0.3, label="$R=1$")
        ax.scatter(rho_subunity[:, 0], rho_subunity[:, 1], color='black', s=8, alpha=0.3, label="$R<1$")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig('./images/xyz_distro/xy_full.png', dpi=300)
        plt.close(fig)

    def reduction_density(reduction_data, density_data, bound = ''):

        reduction_data = np.array(reduction_data)
        density_data = np.array(density_data)
        def stats(n):
            sample_r = []
            for i in range(0, len(density_data)):
                if np.abs(np.log10(density_data[i]/n)) < 1/8:
                    sample_r.append(reduction_data[i])
            sample_r.sort()
            if len(sample_r) == 0:
                return [np.nan, np.nan, np.nan]

            mean   = np.mean(sample_r)
            median = np.quantile(sample_r, .5)
            ten    = np.quantile(sample_r, .1)
            return [mean, median, ten]
            
        mask = reduction_data != 1
        reduction_data = reduction_data[mask]
        density_data = density_data[mask]
        fraction = (mask.shape[0] - np.sum(mask)) / mask.shape[0] # {R = 1}/{R}
        Npoints = len(reduction_data)
        n_min, n_max = np.log10(np.min(density_data)), np.log10(np.max(density_data))
        x_n = np.logspace(n_min, n_max, Npoints)
        mean_vec = np.zeros(Npoints)
        median_vec = np.zeros(Npoints)
        ten_vec = np.zeros(Npoints)
        for i in range(0, Npoints):
            s = stats(x_n[i])
            mean_vec[i] = s[0]
            median_vec[i] = s[1]
            ten_vec[i] = s[2]

                    
        rdcut = []
        for i in range(0,Npoints):
            if density_data[i] > n_min:
                rdcut = rdcut + [reduction_data[i]]


        fig = plt.figure(figsize = (12, 6))
        ax1 = fig.add_subplot(121)
        ax1.hist(rdcut, round(np.sqrt(Npoints)))
        ax1.set_xlabel('Reduction factor', fontsize = 16)
        ax1.set_ylabel('number', fontsize = 16)
        ax1.set_title(f't = {time}', fontsize = 16)
        plt.setp(ax1.get_xticklabels(), fontsize = 16)
        plt.setp(ax1.get_yticklabels(), fontsize = 16)
        ax2 = fig.add_subplot(122)
        l1, = ax2.plot(x_n, mean_vec)
        l2, = ax2.plot(x_n, median_vec)
        l3, = ax2.plot(x_n, ten_vec)
        try:
            ax2.scatter(density_data, reduction_data, alpha = 0.5, color = 'grey')
        except:
            pass
        plt.legend((l1, l2, l3), ('mean', 'median', '10$^{\\rm th}$ percentile'), loc = "lower right", prop = {'size':14.0}, ncol =1, numpoints = 5, handlelength = 3.5)
        plt.xscale('log')
        plt.ylim(0.25, 1.05)
        ax2.set_ylabel('Reduction factor', fontsize = 16)
        ax2.set_xlabel('gas density (hydrogens per cm$^3$)', fontsize = 16)
        ax2.set_title(f'f(R=1) = {fraction}', fontsize = 16)
        plt.setp(ax2.get_xticklabels(), fontsize = 16)
        plt.setp(ax2.get_yticklabels(), fontsize = 16)
        fig.subplots_adjust(left = .1)
        fig.subplots_adjust(bottom = .15)
        fig.subplots_adjust(top = .98)
        fig.subplots_adjust(right = .98)
        fig.tight_layout()

        #plt.savefig('histograms/pocket_statistics_ks.pdf')
        plt.savefig(f'./images/reduction/pocket_stats{bound}.png', dpi=300)

    #reduction_density(r_l, n_rs, 'l')

    reduction_density(r_u, n_rs, 'u')

    if True:
        r_u = np.array(r_u) # threshold * 1
        r_l = np.array(r_l) # threshold * 10

        total = r_u.shape[0]
        mask = r_u == 1.0  
        ones = np.sum(mask)
        fraction = ones/total
        x_ones = x_input[mask,:]
        x_not  = x_input[np.logical_not(mask),:]

        tda(x_input, 'x')
        tda(x_ones, 'R = 1')
        tda(x_not , 'R < 1')

    if True:
        try:
                
            from matplotlib import cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=np.min(magnetic_fields), vmax=np.max(magnetic_fields))
            cmap = cm.viridis

            ax = plt.figure().add_subplot(projection='3d')
            radius_vectors /= pc_to_cm

            for k in range(m):
                if m > 100:
                    break
                # mask makes sure that start and ending point arent the zero vector
                numb_densities[:, k]
                mask = numb_densities[:, k] > 0

                x=radius_vectors[mask, k, 0]
                y=radius_vectors[mask, k, 1]
                z=radius_vectors[mask, k, 2]
                
                for l in range(len(x) - 1):
                    color = cmap(norm(magnetic_fields[l, k]))
                    ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=0.3)

                ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
                ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
                    
            radius_to_origin = np.sqrt(x**2 + y**2 + z**2)
            zoom = (np.max(radius_to_origin) + rloc)/2.0
            ax.set_xlim(-zoom,zoom)
            ax.set_ylim(-zoom,zoom)
            ax.set_zlim(-zoom,zoom)
            ax.set_xlabel('x [Pc]')
            ax.set_ylabel('y [Pc]')
            ax.set_zlabel('z [Pc]')
            ax.set_title('Magnetic field morphology')

            # Add a colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Magnetic Field Strength')
            plt.savefig("./images/FieldTopology.png", bbox_inches='tight', dpi=300)

        except:
            print("Couldnt print B field structure")

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

lines = magnetic_fields.shape[1]

zeta_at_x = np.zeros(lines)
nmir_at_x = np.zeros(lines)

zeta_full = []
nmir_full = []

for line in range(lines):
    #if line <= _exit_:
    #    break
    print("\n",line,"/",lines)
    numb_density    = numb_densities[:, line]
    magnetic_field =  magnetic_fields[:, line]
    radius_vector  =  radius_vectors[:, line, :]
    
    # --- Adjustments / Resizing ---
    mask = np.where(numb_density > 0.0)[0]
    start, end = mask[0], mask[-1]

    #print(start, follow_index//2, end)
    numb_density    = numb_density[start:end]
    magnetic_field =  magnetic_field[start:end]
    radius_vector  =  radius_vector[start:end, :]
    trajectory = np.cumsum(np.linalg.norm(radius_vector, axis=1)) #np.insert(, 0, 0.0)
    
    #magnetic_field =  np.ones_like(magnetic_field)* np.mean(magnetic_field)
    #magnetic_field[magnetic_field.shape[0]//2] *= 1.01

    if follow_index0 > end - start:
        follow_index = follow_index0 - start

    #trajectory = np.cumsum(np.linalg.norm(np.diff(radius_vector, axis=0), axis=1))

    """ Column Densities N_+(mu, s) & N_-(mu, s)"""

    Npmu, mu_local_fwd, dmu_fwd, t_fwd = column_density(radius_vector, magnetic_field, numb_density, "fwd")

    Nmmu, mu_local_bwd, dmu_bwd, t_bwd = column_density(radius_vector[::-1, :], magnetic_field[::-1], numb_density[::-1], "bwd")

    if False:

        for mui in range(Npmu.shape[1]):
            plt.plot(Npmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}")    
            #plt.legend()

        plt.title("Column density $N_+(\mu_i)$  ")
        plt.xlabel("Index")
        plt.ylabel(r"$N_+(\mu_i, s)$")
        #plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'./images/i_rate/Model{model}_forward_column.png', dpi=300)  # Save as PNG with high resolution
        plt.close()

        for mui in range(Npmu.shape[1]):
            plt.plot(Nmmu[:, mui], label=f"$\mu_i = ${mu_ism[mui]}")    
            #plt.legend()

        plt.title("Column density $N_-(\mu_i)$  ")
        plt.xlabel("Index")
        plt.ylabel(r"$N_-(\mu_i, s)$")
        #plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'./images/i_rate/Model{model}_backward_column.png', dpi=300)  # Save as PNG with high resolution
        plt.close()

        for mui in range(dmu_fwd.shape[1]):
            plt.plot(dmu_fwd[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
            #plt.legend()

        plt.title(r"Evolution of $\frac{B}{B_i}\frac{\mu_i}{\mu}$ (Forwards)")
        plt.xlabel("trajectory (cm)")
        plt.ylabel(r"$\frac{\partial \mu_i}{\partial \mu}$")
        plt.xscale('log')
        plt.savefig(f'./images/i_rate/Model{model}_forward_jacobian.png', dpi=300)  # Save as PNG with high resolution
        plt.close()
        
        for mui in range(dmu_bwd.shape[1]):
            plt.plot(dmu_bwd[:, mui], label=f"$\mu_i = ${mu_ism[mui]}") 
            #plt.legend()

        plt.title(r"Evolution of $\frac{B}{B_i}\frac{\mu_i}{\mu}$ (Backwards)")
        plt.xlabel("trajectory (cm)")
        plt.ylabel(r"$\frac{\partial \mu_i}{\partial \mu}$")
        plt.xscale('log')
        plt.savefig(f'./images/i_rate/Model{model}_backward_jacobian.png', dpi=300)  # Save as PNG with high resolution
        plt.close()

    max_arg = np.argmax(magnetic_field)

    #Nmir_fwd = mirrored_column_density(radius_vector[:max_arg,:], magnetic_field[:max_arg], numb_density[:max_arg+1], Npmu[:max_arg,:], 'mir_fwd')
    #Nmir_bwd = mirrored_column_density(radius_vector[max_arg:,:][::-1,:], magnetic_field[max_arg:][::-1], numb_density[max_arg:][::-1], Nmmu[max_arg:,:], 'mir_bwd')

    Nmir_fwd = mirrored_column_density(radius_vector, magnetic_field, numb_density, Npmu, 'mir_fwd')
    Nmir_bwd = mirrored_column_density(radius_vector[::-1,:], magnetic_field[::-1], numb_density[::-1], Nmmu, 'mir_bwd')

    """ Ionization Rate for N = N(s) """

    #zeta_mir_fwd,  zeta_mui_mir_fwd, spectra_fwd  = ionization_rate(Nmir_fwd, mu_local_fwd[:max_arg,:], dmu_fwd[:max_arg,:], 'mir_fwd')
    #zeta_mir_bwd, zeta_mui_mir_bwd, spectra_bwd = ionization_rate(Nmir_bwd, mu_local_bwd[max_arg:,:], dmu_bwd[max_arg:,:], 'mir_bwd')

    zeta_mir_fwd, zeta_mui_mir_fwd, spectra_fwd  = ionization_rate(Nmir_fwd, mu_local_fwd, dmu_fwd, 'mir_fwd')
    zeta_mir_bwd, zeta_mui_mir_bwd, spectra_bwd   = ionization_rate(Nmir_bwd, mu_local_bwd, dmu_bwd, 'mir_bwd')

    Nmir = np.sum(np.concatenate((Nmir_fwd, Nmir_bwd[::-1]), axis=0), axis=1)
    #zeta = np.concatenate((zeta_mir_fwd, zeta_mir_bwd[::-1]), axis=0)
    zeta = (zeta_mir_fwd+ zeta_mir_bwd[::-1])
    
    #print(zeta.shape, zeta_at_x.shape)
    zeta_full += zeta.tolist()
    nmir_full += Nmir.tolist()
    print(trajectory.shape)
    print(zeta.shape)

    if False:
        fig, ax = plt.subplots()

        #ax.scatter(nmir_reduced, zeta_reduced, s=8, marker='x', color="red", label=fr"$\zeta$")
        #ax.plot(trajectory, np.log10(zeta), "-",color="red", label=fr"$\zeta$")
        #ax.plot(trajectory, np.log10(zeta_mir_fwd), ".",color="grey", label=fr"$\zeta_+$", alpha=0.5)
        #ax.plot(trajectory, np.log10(zeta_mir_bwd[::-1]), "--",color="grey", label=fr"$\zeta_-$", alpha=0.5)
        zeta_log = np.log10(zeta)
        ax.plot(trajectory, zeta_log, "-", color="black", label=fr"$\zeta$")
        ax.plot(trajectory,(magnetic_field - magnetic_field.min()) / (magnetic_field.max() - magnetic_field.min()) * ((-16.5) - (-19)) + (-19),"--", ms=3,  color="red", label=fr"$B(s)$")
        #ax.set_title("Total Ionization at X", fontsize=16)
        ax.set_xlabel("Distance (cm)", fontsize=16)
        ax.set_ylabel("$\log_{10}(\zeta / s^{-1})$", fontsize=16)
        #ax.set_xscale('log')
        ax.set_ylim(-19.5, -16)
        fig.tight_layout()
        plt.grid(True)
        plt.savefig(f'./images/i_rate/Model{model}_marginal.png', dpi=300)
        plt.close(fig)

    zeta_at_x[line] = zeta[follow_index]
    nmir_at_x[line] = Nmir[follow_index]


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

if __name__ == '__main__':
    pass
    #re_do_stats_py_plots()

