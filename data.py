"""
Read and Plots statistics from 

./series/r_stats.pkl
./series/c_stats.pkl
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from src.library import *
from scipy.stats import skew
from scipy.stats import kurtosis
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

def dual_log_log(x, y, xlabel, ylabel, ylabels, output) -> None:
    """Plot two log-log scatter plots side by side with consistent styling."""
    y0, y1 = y
    ylabel0, ylabel1 = ylabels

    fig, (ax_l, ax_h) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'wspace': 0},
        sharey=True
    )
    flag = True

    for ax, ydata, label in zip([ax_l, ax_h], [y0, y1], [ylabel0, ylabel1]):
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=GRID_ALPHA)
        # Plot multiple markers for the same series if needed
        for marker, color in zip(MARKERS, COLORS):
            if flag:
                ax.scatter(x, ydata, marker=MARKERS[0], color=color, s=SIZE, alpha=ALPHA, label=label)
                flag=False
        flag=True
        ax.legend(fontsize=FONTSIZE-2)

    ax_l.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax_h.tick_params(labelleft=False)  # Remove y-labels on right plot
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

def log_log(x, y, xlabel, ylabel, output) -> None:
    """Plot single log-log scatter plot with consistent styling."""
    y0 = y[0] if isinstance(y, (list, tuple)) else y

    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=GRID_ALPHA)

    for marker, color in zip(MARKERS, COLORS):
        ax.scatter(x, y0, marker=marker, color=color, s=SIZE, alpha=ALPHA, label=ylabel)
    ax.legend(fontsize=FONTSIZE-2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

def mono_log_log(x, y_list, xlabel, ylabel, labels, output) -> None:
    """
    Plot multiple log-log scatter series on a single plot.

    Parameters:
    - x: array-like, shared x-axis values
    - y_list: list of arrays, each array is a different y series
    - xlabel: str, x-axis label
    - ylabel: str, y-axis label
    - labels: list of str, labels for each y series
    - output: str, filename to save the figure
    """
    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=GRID_ALPHA)

    # Repeat markers and colors if more series than defined
    markers = MARKERS * ((len(y_list) // len(MARKERS)) + 1)
    colors  = COLORS  * ((len(y_list) // len(COLORS)) + 1)

    for ydata, label, marker, color in zip(y_list, labels, markers, colors):
        ax.scatter(x, ydata, marker=marker, color=color, s=SIZE, alpha=ALPHA, label=label)

    ax.legend(fontsize=FONTSIZE-2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

def log_lin(xlabel, labels, ylabel, output) -> None:

    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    # Repeat markers and colors if more series than defined
    markers = MARKERS * ((len(y_list) // len(MARKERS)) + 1)
    colors  = COLORS  * ((len(y_list) // len(COLORS)) + 1)

    for ydata, label, marker, color in zip(y_list, labels, markers, colors):
        ax.scatter(x, ydata, marker=marker, color=color, s=SIZE, alpha=ALPHA, label=label)

    ax.legend(fontsize=FONTSIZE-2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)
    
def unstruct_heatmap(x, y, z, output) -> None:

    import matplotlib.tri as tri

    # Create a triangulation object
    triang = tri.Triangulation(x, y)

    # Plot with tripcolor (heatmap)
    plt.figure(figsize=(6,5))
    tpc = plt.tripcolor(triang, z, shading='gouraud', cmap='plasma')
    plt.colorbar(tpc, label='Z value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Unstructured heatmap via triangulation')
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f'{output}.png', dpi=300)

""" Output:
        stats_dict = {
            "time": _time,
            "x_input": x_input,
            "n_rs": n_rs,
            "B_rs": B_rs,
            "n_path": path_column,
            "n_los0": mean_column,
            "n_los1": median_column,
            "surv_fraction": survivors_fraction[each],
            "r_u": r_u,
            "r_l": r_l
        }
        field_dict = {
            "time": _time,
            "x_input": x_input,
            "B_s": magnetic_fields,
            "r_s": radius_vectors,
            "n_s": numb_densities
        }
"""
if os.path.exists(f'./series/data_{INPUT}.pkl'):
    # Load the pickled DataFrames
    df = pd.read_pickle(f'./series/data_{INPUT}.pkl')
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

if os.path.exists(f'./series/data1_{INPUT}.pkl'):
    df = pd.read_pickle(f'./series/data1_{INPUT}.pkl')
    df.index.name = 'snapshot'
    df.index = df.index.astype(int)
    df = df.sort_index()

    x_input   = df["x_input"].to_numpy()
    directions= df["directions"].to_numpy()
    fields    = df["B_s"].to_numpy()
    densities = df["n_s"].to_numpy()
    vectors   = df["r_s"].to_numpy()

    Neff  = np.logspace(19, 27, 10_000) 
    # (Padovani et al 2018) LLR - Long lived radionuclei  
    log_zeta_llr  = np.log10(1.4e-22)*np.ones_like(Neff) 

    log_zeta_low, log_zeta_high = ionization_rate_fit(Neff)
    n_mirr_at_x, zeta_at_x = ionization_rate(fields, densities, vectors, x_input)




if __name__ == '__main__':
    """
    IONIZATION RATE

    \zeta_i(s) = \int_{-1}^{1}d\mu \int_0^{\infty} j(E', \mu, s) \sigma_{ion}(E')dE'
    
    """
    output = 'ion_lh_'

    log_ionization_path_l, log_ionization_path_h = ionization_rate_fit(Ncrs[-1])
    log_ionization_los_l, log_ionization_los_h   = ionization_rate_fit(Nlos0[-1])

    fig, axs = plt.subplots(2, 2, figsize=(16, 5),gridspec_kw={'wspace': 0, 'hspace': 0}, sharey=True)

    axs[0,0].set_title("Model $\mathcal{L}$", fontsize=16)
    axs[0,0].set_xlabel(r'''$n_g$ (cm$^{-3}$)''', fontsize=16)
    axs[0,0].set_ylabel(r'$log_{10}(\zeta /\rm{s}^{-1})$', fontsize=16)
    axs[0,0].scatter(Nlos0[-1], log_ionization_los_l, marker='o', color="#8E2BAF", s=5, alpha=0.3, label=r'$\zeta^{\mathcal{L}}_{\rm los}$')
    axs[0,0].set_xscale('log')
    axs[0,0].legend(fontsize=16, loc='lower left')
    axs[1,0].scatter(Ncrs[-1], log_ionization_path_l, marker='v', color="#148A02", s=5, alpha=0.3, label=r'$\zeta^{\mathcal{L}}_{\rm path}$')
    axs[1,0].set_xscale('log')
    axs[1,0].grid(True, which='both', alpha=0.3)
    axs[1,0].legend(fontsize=16, loc='lower left')
    axs[1,0].set_ylim(-20, -16)
    axs[0,0].set_ylim(-20, -16)

    axs[0,1].set_title("Model $\mathcal{H}$", fontsize=16)
    axs[0,1].set_xlabel(r'''Column Density [cm$^{-2}$]''', fontsize=16)
    axs[0,1].set_xscale('log')
    axs[0,1].scatter(Nlos0[-1], log_ionization_los_h, marker='o', color="#8E2BAF", s=5, alpha=0.3, label=r'$\zeta^{\mathcal{H}}_{\rm los}$')
    axs[0,1].legend(fontsize=16)
    axs[1,1].set_xscale('log')
    axs[1,1].scatter(Ncrs[-1], log_ionization_path_h, marker='v', color="#148A02", s=5, alpha=0.3, label=r'$\zeta^{\mathcal{H}}_{\rm path}$')
    axs[1,1].grid(True, which='both', alpha=0.3)
    axs[1,1].legend(fontsize=16)
    axs[1,1].set_ylim(-20, -16)
    axs[0,1].set_ylim(-20, -16)
    
    axs[1,1].tick_params(labelleft=False)

    fig.suptitle(r"Ionization Rate (s$^{-1}$) vs. Column Density [cm$^{-2}$]", fontsize=18)
    fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)
    
    exit()
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
    ax1.set_ylabel(r"$\frac{\mathcal{n}_{local}}{\mathcal{n}_{ism}}$ [Adim]", fontsize=FONTSIZE)
    ax1.grid(True, which='both', alpha=GRID_ALPHA)
    ax1.plot(t, f_mean, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(t, f_median, '--', color='black', alpha=ALPHA, label=r'Median')
    ax1.plot(t, f_less, '.-', color='orange', alpha=ALPHA, label=r'$f_{R=1}$')
    ax1.legend(fontsize=FONTSIZE-2)
    percentiles = [1, 5, 10, 20, 25]

    for ptile in percentiles[::-1]:  # plot largest band first for proper layering
        f_ptile_down = np.array([np.percentile(ul[ul<1], ptile) for ul in factu])
        f_ptile_up   = np.array([np.percentile(ul[ul<1], 100-ptile) for ul in factu])
        
        ax1.fill_between(t, f_ptile_down, f_ptile_up,
                        color=COLORS[0], alpha=0.3, label=f'{ptile}–{100-ptile} percentile', zorder=1)

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

    ax1.set_ylim(-0.1,1.1)

    ax2.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax2.grid(True, which='both', alpha=GRID_ALPHA)
    ax2.plot(t, f_skew, '-', color='black', alpha=ALPHA, label=r'$\gamma$ : Skewness $=\frac{E[(X-\mu)^3]}{\sigma^3}$')
    ax2.plot(t, f_kurt, '--', color='black', alpha=ALPHA, label=r'$\kappa$: Kurtosis $= \frac{E[(X-\mu)^4]}{\sigma^4}$')
    ax2.legend(fontsize=FONTSIZE-2)

    # Create inset axes (zoom window)
    axins = inset_axes(ax2, width="40%", height="40%", loc='upper right')  # adjust size and location
    axins.plot(t, f_skew, '-', color='black', alpha=ALPHA)
    axins.plot(t, f_kurt, '--', color='black', alpha=ALPHA)

    x1, x2 = 4.5515, 4.5523
    y1, y2 = min(np.min(f_skew), np.min(f_kurt))*1.5, max(np.max(f_skew), np.max(f_kurt))*1.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, which='both', alpha=0.5)

    fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    f_mean   = np.array([np.mean(ul[ul<1]) for ul in factl])
    f_median = np.array([np.median(ul[ul<1]) for ul in factl])
    f_std    = np.array([np.std(ul[ul<1]) for ul in factl])
    f_skew   = np.array([skew(ul[ul<1]) for ul in factl])
    f_kurt   = np.array([kurtosis(ul[ul<1]) for ul in factl])
    f_less   = np.array([np.sum(ul<1)/ul.shape[0] for ul in factl])

    output = 'rl_'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$\frac{\mathcal{n}_{local}}{\mathcal{n}_{ism}}$ [Adim]", fontsize=FONTSIZE)
    ax1.grid(True, which='both', alpha=GRID_ALPHA)
    ax1.plot(t, f_mean, '-', color='black', alpha=ALPHA, label=r'Mean')
    ax1.plot(t, f_median, '--', color='black', alpha=ALPHA, label=r'Median')
    ax1.plot(t, f_less, '.-', color='orange', alpha=ALPHA, label=r'$f_{R=1}$')
    ax1.legend(fontsize=FONTSIZE-2)
    percentiles = [1, 5, 10, 20, 25]

    for ptile in percentiles[::-1]:  # plot largest band first for proper layering
        f_ptile_down = np.array([np.percentile(ul[ul<1], ptile) for ul in factu])
        f_ptile_up   = np.array([np.percentile(ul[ul<1], 100-ptile) for ul in factu])
        
        ax1.fill_between(t, f_ptile_down, f_ptile_up,
                        color=COLORS[0], alpha=0.3, label=f'{ptile}–{100-ptile} percentile', zorder=1)

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

    ax1.set_ylim(-0.1,1.1)
    ax2.set_xlabel(r"Time [Myrs]", fontsize=FONTSIZE)
    ax2.grid(True, which='both', alpha=GRID_ALPHA)
    ax2.plot(t, f_skew, '-', color='black', alpha=ALPHA, label=r'$\gamma$ : Skewness $=\frac{E[(X-\mu)^3]}{\sigma^3}$')
    ax2.plot(t, f_kurt, '--', color='black', alpha=ALPHA, label=r'$\kappa$: Kurtosis $= \frac{E[(X-\mu)^4]}{\sigma^4}$')
    ax2.legend(fontsize=FONTSIZE-2)
    axins = inset_axes(ax2, width="40%", height="40%", loc='upper right')  # adjust size and location
    axins.plot(t, f_skew, '-', color='black', alpha=ALPHA)
    axins.plot(t, f_kurt, '--', color='black', alpha=ALPHA)

    x1, x2 = 4.5515, 4.5523
    y1, y2 = min(np.min(f_skew), np.min(f_kurt))*1.5, max(np.max(f_skew), np.max(f_kurt))*1.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(True, which='both', alpha=0.5)

    fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    # plot and save Reduction factor over n_g using log10(n_g(p)/n_{ref}) < 1/8

    output = 'r_n_'
    n_plots = 20

    fig, ax = plt.subplots()

    ax.set_xlabel(r"$n_g$ [cm$^{-3}$]", fontsize=FONTSIZE)
    ax.set_ylabel(r"$1 - \mathcal{N}_{local}/ \mathcal{N}_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(t[:n_plots]), max(t[:n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(t[:n_plots], factu[:n_plots])):
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val, n[i])
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, 1 - mean_vec, '-', lw=2.0, color=color, alpha=0.6)

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
    ax.set_ylabel(r"$\mathcal{N}_{local}/ \mathcal{N}_{ism}$ [Adim]", fontsize=FONTSIZE)
    ax.grid(True, which='both', alpha=GRID_ALPHA)
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)

    cmap = plt.get_cmap("viridis")
    t_min, t_max = min(t[:n_plots]), max(t[:n_plots])

    # Plot lines with colors
    for i, (t_val, rt_val) in enumerate(zip(t[:n_plots], factu[:n_plots])):
        n_ref, r_matrix, mean_vec, median_vec, ten_vec, sample_size = reduction_to_density(rt_val, n[i])
        normalized_t = (t_val - t_min) / (t_max - t_min)        
        color = cmap(normalized_t)
        ax.plot(n_ref, mean_vec, '-', lw=2.0, color=color, alpha=0.6)

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
    # Column densities
    print(Nlos0.shape,Nlos1.shape,Ncrs.shape)
    print(len(Nlos0[0]),len(Nlos0[1]), len(Nlos0[2]))
    print(len(Nlos1[0]),len(Nlos1[1]), len(Nlos1[2]))
    #
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

    fig.tight_layout()
    plt.savefig('series/' + output + f'{INPUT}.png', dpi=300)
    plt.close(fig)

    xlabel = r"$N_{crs}$ [cm$^{-2}$]"
    ylabel = r"$\frac{\mathcal{N}_{local}}{\mathcal{N_{ISM}}}$" 
    labels = [r"$n_{th} = 10^2$ [cm$^-3$]", r"$n_{th} = 10$ [cm$^-3$]"]
    y_list = [factu[-1][factu[-1]<1], factl[-1][factl[-1]<1]]
    x      = Ncrs[-1][factu[-1]<1]
    output = f"fu_crs"

    log_lin(xlabel=xlabel, labels=labels, ylabel=ylabel, output=output)

    # plot and save n^mean_los / n_path and n^median_los / n_path
    # plot and save n^mean_los / n_path and n^median_los / n_path

    y_list = [Nlos0[-1], Nlos1[-1]]
    xlabel =  "$N_{crs}$ [cm$^{-2}$]"
    ylabel =  "Column Density [cm$^{-2}$]"
    labels = [r"$ \langle N_{los} \rangle $ [cm$^{-2}$]", r"$N_{los}^{50th}$ [cm$^{-2}$]"]

    dual_log_log(Ncrs[-1], y_list, xlabel=xlabel, ylabels=labels, ylabel=ylabel, output="multi")
    mono_log_log(Ncrs[-1], y_list ,xlabel=xlabel, labels=labels, ylabel=ylabel, output="mono")


    """
    LOCAL COSMIC RAY SPECTRA

    j(E, \mu, s) = \frac{j_i(E_i, \mu_i, N(s))L(E_i)}{2L(E)}
    
    """

    