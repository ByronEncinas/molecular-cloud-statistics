"""
Read and Plots statistics from 

./series/r_stats.pkl
./series/c_stats.pkl
"""
from matplotlib import cm
from matplotlib.colors import Normalize, LogNorm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
from src.library import *
from scipy.stats import skew
from scipy.stats import kurtosis
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

mpl.rcParams['text.usetex'] = True
# Constants for consistent style
MARKERS = ['v', 'o']
COLORS  = ["#8E2BAF", "#148A02"]
ALPHA   = 0.9
SIZE    = 8
FONTSIZE = 12
GRID_ALPHA = 0.5
INPUT = 'ideal'

def field_lines_r_vol(b, r, r0, lr):
    from gists.__vor__ import traslation_rotation
    from copy import deepcopy
    # b   fields
    # r   vectors
    # r0  x_input

    # select line
    m = int(sys.argv[-1]) 
    zoom = 20
    field = b[:,m]
    vector = r[:,m,:]

    # track
    x0, y0, z0 = r0[m]
    x, y, z = vector[:,0] - x0, vector[:,1]-y0, vector[:,2]-z0

    mk = np.logical_and(field > 0, x**2 + y**2 + z**2 < (zoom)**2)
    x, y, z = x[mk], y[mk], z[mk]
    field = field[mk] 
    vector = vector[mk,:]

    bhat = field/np.max(field)

    fig, axd = plt.subplot_mosaic([["profile", "field3d"]], figsize=(10, 4))
    arg_input = np.where(vector[:,0] == r0[m,0])[0][0]
    ax3d = fig.add_subplot(122, projection="3d")
    
    norm = LogNorm(vmin=np.min(bhat), vmax=np.max(bhat))
    cmap = cm.viridis

    for l in range(len(x) - 1):
        color = cmap(norm(bhat[l]))
        ax3d.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=2)
    
    ax3d.scatter(0,0,0, marker="x", color="g", s=6)
    dc = np.array([np.diff(x[arg_input:arg_input+2]), np.diff(y[arg_input:arg_input+2]),np.diff(z[arg_input:arg_input+2])])
    dc /= np.linalg.norm(dc, axis=0) * 1.e+2

    n = 15
    a_ =  6.e-3
    b_ = -a_

    X, Y, Z = np.meshgrid(np.linspace(a_,b_, n), np.linspace(a_,b_, n), np.linspace(a_,b_, n))
    mk = (X**2 + Y**2 + Z**2) < a_**2
    X, Y, Z = X[mk], Y[mk], Z[mk]
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    _x = np.array([0.0, 0.0, 0.0])
    _b = deepcopy(dc)
    p = traslation_rotation(_x, _b[:,0], points)
    ax3d.scatter(p[:,0], p[:,1], p[:,2], marker='x', c='g', alpha=0.3, s=2)

    ax3d.view_init(elev=90, azim=00)
    #ax3d.set_xlim(-0.01,0.01)
    #ax3d.set_ylim(-0.01,0.01)
    #ax3d.set_zlim(-0.01,0.01)
    ax3d.set_xlabel("x [pc]")
    ax3d.set_ylabel("y [pc]")
    ax3d.set_zlabel("z [pc]")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, shrink=0.8)
    cbar.set_label("$\mu$-G")
    field = b[:,m]
    __ = field>0
    field = field[__]
    vector = r[__,m,:]
    displacement = np.cumsum(np.linalg.norm(np.diff(vector, axis=0), axis=1))
    arg_input = np.where(vector[:,0] == r0[m,0])[0][0]
    ax = axd["profile"]
    ax.set_xlabel("Displacement")
    ax.set_ylabel("B [$\mu$G]")
    ax.plot(displacement, field[1:], "-", label="B")

    ax.scatter(displacement[arg_input+1], field[arg_input],c='black',  label="x")
    ax.set_xlim(displacement[arg_input-600], displacement[arg_input+600])
    ax.set_yscale('log')
    ax.legend()
    

    plt.tight_layout()
    plt.savefig("./field_mosaic.png", dpi=300)
    plt.close(fig)
    
def field_lines_norm(b, r, r0):
    
    m = r.shape[1]
    
    elevation = 0
    azimuth   = 0
    zoom      = 0.1     # axis zoom
    zoom2     = 2.0*zoom # spherical window zoom
    """
    r  /= AU_to_cm
    r0 /= AU_to_cm
    r_rxb_z = []
    for k in range(m):
        x0=r0[k, 0]
        y0=r0[k, 1]
        z0=r0[k, 2]
        x=r[:,k, 0]
        y=r[:,k, 1]
        z=r[:,k, 2]
        mk0  = b[:, k] > 0
        mk1  = (x*x + y*y + z*z) < (zoom2)**2
        mk = np.logical_and(mk0,mk1)
        diff_ = np.diff(vectors[mk,k,:], axis=0)
        rxb = np.cross(vectors[mk,k,:][1:,:], diff_, axis=1)
        r_rxb = np.cross(vectors[mk,k,:][1:,:], rxb, axis=1)
        r_rxb_z += [np.mean(r_rxb,axis=0)]
        
    r_rxb_z = np.mean(r_rxb_z, axis=0)*0.1/np.linalg.norm(np.mean(r_rxb_z, axis=0))

    from gists.__vor__ import traslation_rotation

    n = 50
    a_ =  zoom2*0.8
    b_ = -a_

    X, Y = np.meshgrid(np.linspace(a_,b_, n), np.linspace(a_,b_, n)); Z = np.zeros_like(X)
    mk = (X**2 + Y**2) < a_**2 ; X, Y, Z = X[mk], Y[mk], Z[mk]
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    rot_plane = traslation_rotation(np.array([0.0, 0.0, 0.0]), r_rxb_z, points, p=False)
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize, LogNorm

    norm = Normalize(vmin=np.min(b), vmax=np.max(b))
    cmap = cm.viridis

    ax = plt.figure().add_subplot(projection='3d')
    
    for k in range(m):
        x0=r0[k, 0]
        y0=r0[k, 1]
        z0=r0[k, 2]
        ax.scatter(x0, y0, z0, marker="x",color="g",s=6)            
            
    #ax.scatter(rot_plane[:,0], rot_plane[:,1], rot_plane[:,2], marker="x",color="r",s=3, alpha=0.25)            
    #ax.quiver(0,0,0, r_rxb_z[0],r_rxb_z[1],r_rxb_z[2],color="black")

    ax.set_xlim(-zoom*2,zoom*2)
    ax.set_ylim(-zoom*2,zoom*2)
    ax.set_zlim(-zoom*2,zoom*2)
    ax.set_xlabel('x [au]')
    ax.set_ylabel('y [au]')
    ax.set_zlabel('z [au]')
    ax.set_title('Starting Points')
    ax.view_init(elev=elevation, azim=azimuth)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Arbitrary Units')
    plt.savefig("./StartingPoints.png", bbox_inches='tight', dpi=300)

    try:

        from matplotlib import cm
        from matplotlib.colors import Normalize

        ax = plt.figure().add_subplot(projection='3d')
        #r /= pc_to_cm

        for k in range(0,m,1):
            # mask makes sure that start and ending point arent the zero vector
            x0=r0[k, 0]
            y0=r0[k, 1]
            z0=r0[k, 2]
            ax.scatter(x0, y0, z0, marker="x",color="black",s=1,alpha=0.05)   
            x=r[:,k, 0]
            y=r[:,k, 1]
            z=r[:,k, 2]
            mk0  = b[:, k] > 0
            mk1  = x*x + y*y + z*z < (zoom2)**2
            mk = np.logical_and(mk0,mk1)
            x=r[mk,k, 0]
            y=r[mk,k, 1]
            z=r[mk,k, 2]
            bhat = b[mk, k]
            bhat /= np.max(bhat)
            norm = LogNorm(vmin=np.min(bhat), vmax=np.max(bhat))
            cmap = cm.viridis

            ax.scatter(x0, y0, z0, marker="x",color="g",s=1, alpha=0.5, label="X")            
            for l in range(len(x) - 1):
                color = cmap(norm(bhat[l]))
                ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=0.3)

        #ax.scatter(rot_plane[:,0], rot_plane[:,1], rot_plane[:,2], marker="x",color="r",s=3, alpha=0.25)            
        #ax.quiver(0,0,0, r_rxb_z[0],r_rxb_z[1],r_rxb_z[2],color="black")
        ax.set_xlim(-zoom*2,zoom*2)
        ax.set_ylim(-zoom*2,zoom*2)
        ax.set_zlim(-zoom*2,zoom*2)
        ax.set_xlabel('x [Pc]')
        ax.set_ylabel('y [Pc]')
        ax.set_zlabel('z [Pc]')
        ax.set_title('Magnetic field morphology')
        ax.view_init(elev=elevation, azim=azimuth)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Arbitrary Units')
        

        plt.savefig("./FieldTopology.png", bbox_inches='tight', dpi=300)

    except Exception as e:
        print(e)
        print("Couldnt print B field structure")

def cr_density_on_diff_cloud_densities():

    # Load the pickled DataFrames
    df2 = pd.read_pickle(f'./series/data_dc2.0_{INPUT}.pkl')
    df2.index.name = 'snapshot'
    df2.index = df2.index.astype(int)
    df2 = df2.sort_index()

    df4 = pd.read_pickle(f'./series/data_dc4.0_{INPUT}.pkl')
    df4.index.name = 'snapshot'
    df4.index = df4.index.astype(int)
    df4 = df4.sort_index()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    # import n-arrays 
    t2     = df2["time"].to_numpy()
    factu2 = df2["r_u"].to_numpy()
    n2     = df2["n_rs"].to_numpy()
    #x2     = df2["x_input"].to_numpy()
    #B2     = df2["B_rs"].to_numpy()
    #N2los0 = df2["n_los0"].to_numpy()  # mean
    #N2los1 = df2["n_los1"].to_numpy()  # median
    #N2crs  = df2["n_path"].to_numpy()
    #surf2  = df2["surv_fraction"].to_numpy()
    #factl2 = df2["r_l"].to_numpy()

    t4     = df4["time"].to_numpy()
    factu4 = df4["r_u"].to_numpy()
    n4     = df4["n_rs"].to_numpy()
    #x4     = df4["x_input"].to_numpy()
    #B4     = df4["B_rs"].to_numpy()
    #N4los0 = df4["n_los0"].to_numpy()  # mean
    #N4los1 = df4["n_los1"].to_numpy()  # median
    #N4crs  = df4["n_path"].to_numpy()
    #surf4  = df4["surv_fraction"].to_numpy()
    #factl4 = df4["r_l"].to_numpy()

    n_plots = 1

    fig, ax = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'wspace': 0},
        sharey=True
    )

    #fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    output = 'r_n_2_4_'

    cmap = plt.get_cmap("viridis")
    t_min, t_max = 2.9, max(np.max(t4), np.max(t2))
    print(t2.shape, t4.shape)
    print(min(t2), max(t2))
    print(min(t4), max(t4))


    # ---- LEFT PANEL ----
    ax[0].set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=16)
    ax[0].set_title(r"$n_{cloud} > 10^4$ cm$^{-3}$", fontsize=16)
    ax[0].set_ylabel(r"$n_{local}/ n_{ism}$", fontsize=16)
    ax[0].grid(True, which='both', alpha=0.3)
    ax[0].set_xscale('log')
    ax[0].set_ylim(-0.1, 1.1)
    ax[0].tick_params(axis='both', labelsize=14)

    for i, (t_val, n_val ,rt_val) in enumerate(zip(t4[::n_plots], n4[::n_plots], factu4[::n_plots])):
        _ = rt_val < 1.
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val[_], n_val[_])
        color = cmap((t_val - t_min) / (t_max - t_min))
        ax[0].plot(n_ref, mean_vec, '-', lw=2.0, color=color, alpha=0.6)


    ax[1].set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=16)
    ax[1].set_title(r"$n_{cloud} > 10^2$ cm$^{-3}$", fontsize=16)
    ax[1].grid(True, which='both', alpha=0.3)
    ax[1].set_xscale('log')
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].tick_params(axis='both', labelsize=14)

    for i, (t_val, n_val ,rt_val) in enumerate(zip(t2[::n_plots], n2[::n_plots], factu2[::n_plots])):
        _ = rt_val < 1.
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val[_], n_val[_])
        color = cmap((t_val - t_min) / (t_max - t_min))
        ax[1].plot(n_ref, mean_vec, '-', lw=2.0, color=color, alpha=0.6)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.set_label('Time [Myrs]', fontsize=16)
    cbar.ax.tick_params(labelsize =14)

    plt.savefig('./' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)
    return None

def local_cr_spectra(size):
    energy = np.logspace(3, 9, size)
    C, alpha, beta, Enot = select_species('L')
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))/(4*np.pi)
    log_spec_ism_low  = np.log10(ism_spectrum(energy))
    C, alpha, beta, Enot = select_species('H')
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))/(4*np.pi)
    log_spec_ism_high = np.log10(ism_spectrum(energy))
    log_energy = np.log10(energy)

    
    """
    IONIZATION RATE

    \zeta_i(s) = \int_{-1}^{1}d\mu \int_0^{\infty} j(E', \mu, s) \sigma_{ion}(E')dE'
    
    """
    Neff  = np.logspace(19, 27, size) 

    # (Padovani et al 2018) LLR - Long lived radionuclei  
    log_zeta_llr  = np.log10(1.4e-22)*np.ones_like(Neff) 
    log_zeta_std  = np.log10(1.0e-17)*np.ones_like(Neff) 
    log_zeta_low, log_zeta_high = ionization_rate_fit(Neff)
    loss_function = lambda z: Lstar*(Estar/z)**d 
    n_mirr_l_at_x, zeta_l_at_x, loc_spec_l_at_x = x_ionization_rate(fields[0], densities[0], vectors[0], x_input[0], m='L')
    n_mirr_h_at_x, zeta_h_at_x, loc_spec_h_at_x = x_ionization_rate(fields[0], densities[0], vectors[0], x_input[0], m='H')
    

    output = 'ion_lh_'
    
    log_ionization_path_l, log_ionization_path_h = ionization_rate_fit(Ncrs[-1])
    log_ionization_los_l, log_ionization_los_h   = ionization_rate_fit(Nlos0[-1])

    fig, axs = plt.subplots(1, 2, figsize=(5, 5),gridspec_kw={'wspace': 0, 'hspace': 0}, sharey=True)

    _x = min(np.min(Nlos0[-1]), np.min(Ncrs[-1]))*15
    _y = -16.4

    axs[0].text(_x, _y, "$\mathcal{L}$",
        fontsize=20,
        color="black",
        rotation=0,
        ha="center",   # horizontal alignment: left, center, right
        va="bottom") # vertical alignment: top, center, bottom
    #axs[0].set_title("Model $\mathcal{L}$", fontsize=16)
    axs[0].set_xlabel(r'''N [cm$^{-2}$]''', fontsize=16)
    axs[0].set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
    axs[0].set_xscale('log')
    axs[0].scatter(Nlos0[-1], log_ionization_los_l, marker='x', color="#8E2BAF", s=15, alpha=0.6, label=r'$\zeta_{\rm los}$')
    axs[0].scatter(Ncrs[-1], log_ionization_path_l-1, marker='|', color="#148A02", s=15, alpha=0.6, label=r'$\zeta_{\rm path}/10$')
    axs[0].axhline(y=log_zeta_std[0], linestyle='--', color='black', alpha=0.6)
    axs[0].axhline(y=log_zeta_llr[0], linestyle='--', color='black', alpha=0.6)

    axs[0].grid(True, which='both', alpha=0.3)
    axs[0].set_ylim(-22, -16)

    _x = min(np.min(Nlos0[-1]), np.min(Ncrs[-1]))*15

    axs[1].text(_x, _y, "$\mathcal{H}$",
        fontsize=20,
        color="black",
        rotation=0,
        ha="center",   # horizontal alignment: left, center, right
        va="bottom") # vertical alignment: top, center, bottom
    
    axs[1].set_xlabel(r'''$N$ [cm$^{-2}$]''', fontsize=16)
    axs[1].set_xscale('log')
    axs[1].scatter(Nlos0[-1], log_ionization_los_h, marker='x', color="#8E2BAF", s=15, alpha=0.6, label=r'$\zeta_{\rm los}$')
    axs[1].scatter(Ncrs[-1], log_ionization_path_h-1, marker='|', color="#148A02", s=15, alpha=0.6, label=r'$\zeta_{\rm path}/10$')
    axs[1].axhline(y=log_zeta_std[0], linestyle='--', color='black', alpha=0.6)
    axs[1].axhline(y=log_zeta_llr[0], linestyle='--', color='black', alpha=0.6)
    axs[1].grid(True, which='both', alpha=0.3)
    axs[1].set_ylim(-22, -16)
    axs[1].tick_params(labelleft=False)
    axs[1].legend(fontsize=14, loc='lower left')
    #fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    output = 'ion_ratio_lh_'

    log_ionization_path_l, log_ionization_path_h = ionization_rate_fit(Ncrs[-1])
    log_ionization_los_l, log_ionization_los_h   = ionization_rate_fit(Nlos0[-1])

    ratio_los_path_l = log_ionization_los_l - log_ionization_path_l
    ratio_los_path_h = log_ionization_los_h - log_ionization_path_h

    fig, axs = plt.subplots()

    _x = min(np.min(Nlos0[-1]), np.min(Ncrs[-1]))*15
    _y = -16.4

    axs.text(_x, _y, "$\mathcal{L}$",
        fontsize=20,
        color="black",
        rotation=0,
        ha="center",   # horizontal alignment: left, center, right
        va="bottom") # vertical alignment: top, center, bottom
    
    axs.set_xlabel(r'''$N$ [cm$^{-2}$]''', fontsize=16)
    axs.set_ylabel(r'$\zeta(N_{los})/\zeta(N_{crs})$', fontsize=16)
    axs.set_xscale('log')
    axs.scatter(Nlos0[-1], ratio_los_path_l, marker='x', color="#8E2BAF", s=15, alpha=0.6, label=r'Model $\mathcal{L}$')
    axs.scatter(Ncrs[-1], ratio_los_path_h, marker='x', color="#148A02", s=15, alpha=0.6, label=r'Model $\mathcal{H}$')
    axs.grid(True, which='both', alpha=0.3)
    #fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

#if (os.path.exists(f'./series/data_dc2.0_{INPUT}.pkl') and (os.path.exists(f'./series/data_dc4.0_{INPUT}.pkl'))):
    #cr_density_on_diff_cloud_densities()

#file = f'./series/data_{INPUT}.pkl'
#file = f'./series/data1_{INPUT}.pkl'
file = f'./series/data_dc2.0_{INPUT}.pkl'

if os.path.exists(file):
    # Load the pickled DataFrames
    df = pd.read_pickle(file)
    df.index.name = 'snapshot'
    df.index = df.index.astype(int)
    df = df.sort_index()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    # import n-arrays 
    t     = df["time"].to_numpy()
    x     = df["x_input"].to_numpy()
    n     = df["n_rs"].to_numpy()
    B     = df["B_rs"].to_numpy()
    Nlos0 = df["n_los0"].to_numpy()  # mean
    Nlos1 = df["n_los1"].to_numpy()  # median
    Ncrs  = df["n_path"].to_numpy()
    surf  = df["surv_fraction"].to_numpy()
    factu = df["r_u"].to_numpy()
    factl = df["r_l"].to_numpy()

if False: #os.path.exists(f'./series/data1_{INPUT}.pkl'):
    df = pd.read_pickle(f'./series/data1_{INPUT}.pkl')
    print(f'./series/data1_{INPUT}.pkl')
    df.index.name = 'snapshot'
    df.index = df.index.astype(int)
    df = df.sort_index()

    x_input   = df["x_input"].to_numpy()[-1]#*pc_to_cm
    directions= df["directions"].to_numpy()[-1]
    fields    = df["B_s"].to_numpy()[-1]
    densities = df["n_s"].to_numpy()[-1]
    vectors   = df["r_s"].to_numpy()[-1]#*pc_to_cm
    size = 1000 #x_input.shape[0]
    print(f"./series/data1_{INPUT}.pkl")

    args = np.where(factu[-1] < 1.)[0]
    #field_lines_r_vol(fields, vectors, x_input, args)
    #field_lines_norm(fields, vectors, x_input)
    
    """
    LOCAL COSMIC RAY SPECTRA

    j(E, \mu, s) = \frac{j_i(E_i, \mu_i, N(s))L(E_i)}{2L(E)}
    
    """
    #local_cr_spectra(size)
   
if __name__ == '__main__':

    """
    REDUCTION FACTOR

    \frac{\mathcal{n}_{local}}{\mathcal{n}_{ism}} = 1 - \sqrt{1-\frac{B(s)}{B_i}}
    
    """

    # Plot factu and factl evolution in time and maybe a window to the last values
    f_mean   = np.array([np.mean(ul[ul<1]) for ul in factu])
    f_median = np.array([np.median(ul[ul<1]) for ul in factu])
    f_std    = np.array([np.std(ul[ul<1]) for ul in factu])
    f_skew   = np.array([skew(ul[ul<1]) for ul in factu])
    f_kurt   = np.array([kurtosis(ul[ul<1]) for ul in factu])
    f_less   = np.array([np.sum(ul<1)/ul.shape[0] for ul in factu])

    output = 'ru_'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$n_{local}/n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax1.grid(True, which='both', alpha=GRID_ALPHA)
    ax1.plot(t, f_mean, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(t, f_median, '--', color='black', alpha=ALPHA, label=r'Median')
    ax1.plot(t, f_less, '.-', color='orange', alpha=ALPHA, label=r'$f_{R<1}$')
    ax1.legend(fontsize=FONTSIZE-2)

    percentiles = [1, 5, 10, 20, 25]

    for ptile in percentiles[::-1]:  # plot largest band first for proper layering
        f_ptile_down = np.array([np.percentile(ul[ul<1], ptile) for ul in factu])
        f_ptile_up   = np.array([np.percentile(ul[ul<1], 100-ptile) for ul in factu])
        
        ax1.fill_between(t, f_ptile_down, f_ptile_up,
                        color=COLORS[0], alpha=0.3, label=f'{ptile}–{100-ptile} percentile', zorder=1)
        try:
            ax1.text(t[4], f_ptile_up[4], f"P{100-ptile}",
                fontsize=6,
                color="black",
                alpha=1.0,
                rotation=0,
                ha="left",   # horizontal alignment: left, center, right
                va="bottom") # vertical alignment: top, center, bottom

            ax1.text(t[4], f_ptile_down[4], f"P{ptile}",
                fontsize=6,
                color="black",
                alpha=1.0,
                rotation=0,
                ha="left",   # horizontal alignment: left, center, right
                va="bottom") # vertical alignment: top, center, bottom
        except:
            print("[Exists Error] t[4] non existent")
        
    ax1.set_ylim(-0.1,1.1)
    """
    if INPUT =='ideal':
        t = t*1_000 # from Myrs to kyrs
        ax1.set_xlim(t[8]*0.9,t[-1]*1.2)
        ax2.set_xlim(t[8]*0.9,t[-1]*1.2)
    else:
        t = t*1_000 # from Myrs to 0.1 kyrs
        #ax1.set_xlim(t[9],t[-1]+2.0e-6)
        ax2.set_xlim(t[9],t[-1]+2.0e-6)       
    """
    ax2.set_xlabel(r"Time [kyrs]", fontsize=FONTSIZE)
    ax2.grid(True, which='both', alpha=GRID_ALPHA)
    ax2.plot(t, f_skew, '-', color='black', alpha=ALPHA, label=r'$\gamma$ : Skewness')
    ax2.plot(t, f_kurt, '--', color='black', alpha=ALPHA, label=r'$\kappa$: Kurtosis')
    ax2.legend(fontsize=FONTSIZE-2)

    axins = inset_axes(ax2, width="40%", height="40%", loc='upper right')  # adjust size and location
    axins.plot(t, f_skew, '-', color='black', alpha=ALPHA)
    axins.plot(t, f_kurt, '--', color='black', alpha=ALPHA)

    if INPUT =='ideal':
        x1, x2 = 4.5515, 4.5521
    else:
        x1, x2 = 4.2904, 4.2905

    y1, y2 = min(np.min(f_skew), np.min(f_kurt))*1.5, max(np.max(f_skew), np.max(f_kurt))*1.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, which='both', alpha=0.5)

    #fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    f_mean   = np.array([np.mean(ul[ul<1]) for ul in factl])
    f_median = np.array([np.median(ul[ul<1]) for ul in factl])
    f_std    = np.array([np.std(ul[ul<1]) for ul in factl])
    f_skew   = np.array([skew(ul[ul<1]) for ul in factl])
    f_kurt   = np.array([kurtosis(ul[ul<1]) for ul in factl])
    f_less   = np.array([np.sum(ul<1)/ul.shape[0] for ul in factl])
    f_equal  = np.array([np.sum(ul==1)/ul.shape[0] for ul in factl])

    output = 'rl_'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$n_{local}/n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax1.grid(True, which='both', alpha=GRID_ALPHA)
    ax1.plot(t, f_mean, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(t, f_median, '--', color='black', alpha=ALPHA, label=r'Median')
    ax1.plot(t, f_less, '.-', color='orange', alpha=ALPHA, label=r'$f_{R<1}$')
    ax1.legend(fontsize=FONTSIZE-2)
    percentiles = [1, 5, 10, 20, 25]

    for ptile in percentiles[::-1]:  # plot largest band first for proper layering
        f_ptile_down = np.array([np.percentile(ul[ul<1], ptile) for ul in factu])
        f_ptile_up   = np.array([np.percentile(ul[ul<1], 100-ptile) for ul in factu])
        
        ax1.fill_between(t, f_ptile_down, f_ptile_up,
                        color=COLORS[0], alpha=0.3, label=f'{ptile}–{100-ptile} percentile', zorder=1)
        try:
            ax1.text(t[4], f_ptile_up[4], f"P{100-ptile}",
                fontsize=6,
                color="black",
                alpha=1.0,
                rotation=0,
                ha="left",   # horizontal alignment: left, center, right
                va="bottom") # vertical alignment: top, center, bottom

            ax1.text(t[4], f_ptile_down[4], f"P{ptile}",
                fontsize=6,
                color="black",
                alpha=1.0,
                rotation=0,
                ha="left",   # horizontal alignment: left, center, right
                va="bottom") # vertical alignment: top, center, bottom
        except:
            print("[Exists Error] t[4] non existent")

    ax1.set_ylim(-0.1,1.1)
    ax2.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax2.grid(True, which='both', alpha=GRID_ALPHA)
    ax2.plot(t, f_skew, '-', color='black', alpha=ALPHA, label=r'$\gamma$ : Skewness') # (Fisher) \kappa = 0 is normal
    """
    Interpretation:
    Positive skew: The majority of data points are closer to the minimum.
    Negative skew: The majority of data points are closer to the maximum.
    Skewness of 0: The data is perfectly symmetric around the mean
    """
    ax2.plot(t, f_kurt, '--', color='black', alpha=ALPHA, label=r'$\kappa$: Kurtosis') 
    """
    Interpretation of Fisher’s (Excess) Kurtosis:

    Excess kurtosis > 0: The distribution has heavier tails than a normal distribution.
    Excess kurtosis < 0: The distribution has lighter tails than a normal distribution.
    Excess kurtosis = 0: The distribution has the same tail behavior as a normal distribution.
    """
    ax2.legend(fontsize=FONTSIZE-2)
    axins = inset_axes(ax2, width="40%", height="40%", loc='upper right')  # adjust size and location
    axins.plot(t, f_skew, '-', color='black', alpha=ALPHA)
    axins.plot(t, f_kurt, '--', color='black', alpha=ALPHA)

    x1, x2 = 4.5515, 4.5523
    y1, y2 = min(np.min(f_skew), np.min(f_kurt))*1.5, max(np.max(f_skew), np.max(f_kurt))*1.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, which='both', alpha=0.5)

    #fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    # plot and save Reduction factor over n_g using log10(n_g(p)/n_{ref}) < 1/8

    output = 'r_n_'
    n_plots = 1

    fig, ax = plt.subplots()

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$1 - n_{local}/ n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(t[::n_plots]), max(t[::n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(t[::n_plots], factu[::n_plots])):
        _l_ = rt_val < 1
        rt_val = rt_val[_l_]
        n_val = n[i][_l_]
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val, n_val)
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, 1 - mean_vec, '-', lw=2.0, color=color, alpha=1.0)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time [Myrs]', fontsize=FONTSIZE)

    fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$n_{local}/ n_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(t[::n_plots]), max(t[::n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(t[::n_plots], factu[::n_plots])):

        _l_ = rt_val < 1
        rt_val = rt_val[_l_]
        n_val = n[i][_l_]
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val, n_val)
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, mean_vec, '-', lw=2, color=color, alpha=0.6)

    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time [Myrs]', fontsize=FONTSIZE)

    fig.tight_layout()
    plt.savefig('series/' + output + f'2{INPUT}.png', dpi=300)
    plt.close(fig)

    """
    COLUMN DENSITIES

    \int_0^s \frac{n_g(s')ds'}{\hat{\mu}}
    
    """
    ratio0 = np.array([np.mean(nlos0/ncrs) for (nlos0,ncrs) in zip(Nlos0,Ncrs)]) 
    ratio1 = np.array([np.median(nlos1/ncrs) for (nlos1,ncrs) in zip(Nlos1,Ncrs)]) 
    ratio2 = np.array([np.std(nlos0/ncrs) for (nlos0,ncrs) in zip(Nlos0,Ncrs)]) 

    output = 'los_crs_'

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$N_{los}/N_{crs}$ [Adim]", fontsize=FONTSIZE)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', alpha=GRID_ALPHA)

    ax1.plot(t, ratio0, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(t, ratio1, '--', color='black', alpha=ALPHA, label=r'Median')

    ax1.legend(fontsize=FONTSIZE-2)
    percentiles = [1, 5, 10, 20, 25]

    for ptile in percentiles[::-1]:  # plot largest band first for proper layering
        f_ptile_down = np.array([np.percentile(nlos0/ncrs, ptile) for (nlos0,ncrs) in zip(Nlos0,Ncrs)])
        f_ptile_up   = np.array([np.percentile(nlos0/ncrs, 100-ptile) for (nlos0,ncrs) in zip(Nlos0,Ncrs)])
        
        ax1.fill_between(t, f_ptile_down, f_ptile_up,
                        color=COLORS[0], alpha=0.3, label=f'{ptile}–{100-ptile} percentile', zorder=1)
        """
        ax1.text(t[4], f_ptile_up[4], f"P{100-ptile}",
            fontsize=6,
            color="black",
            alpha=1.0,
            rotation=0,
            ha="left",   # horizontal alignment: left, center, right
            va="bottom") # vertical alignment: top, center, bottom

        ax1.text(t[4], f_ptile_down[4], f"P{ptile}",
            fontsize=6,
            color="black",
            alpha=1.0,
            rotation=0,
            ha="left",   # horizontal alignment: left, center, right
            va="bottom") # vertical alignment: top, center, bottom
        """
    fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

