import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

def gkde_plt(x_input, snap):
    os.makedirs("./iso/", exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter(x_input[:,1], x_input[:,2], s=1, alpha=0.3)

    # Compute KDE
    xy = np.vstack([x_input[:,0], x_input[:,1]])
    kde = gaussian_kde(xy)

    # Build a grid and evaluate the KDE on it
    x_grid = np.linspace(x_input[:,0].min(), x_input[:,0].max(), 200)
    y_grid = np.linspace(x_input[:,1].min(), x_input[:,1].max(), 200)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Overlay isocontours
    ax.contour(xx, yy, zz, levels=10, cmap='viridis')

    plt.savefig(f"./iso/iso{snap}.png", dpi = 150)
    plt.close(fig)

def mimic_b68():

    np.random.seed(42)

    # --- Grid ---
    res = 300
    x = np.linspace(-130, 130, res)   # RA offset (arcsec)
    y = np.linspace(-130, 130, res)   # Dec offset (arcsec)
    xx, yy = np.meshgrid(x, y)

    def gauss2d(cx, cy, sx, sy, amp, angle=0):
        a = np.radians(angle)
        c, s = np.cos(a), np.sin(a)
        xr =  c*(xx-cx) + s*(yy-cy)
        yr = -s*(xx-cx) + c*(yy-cy)
        return amp * np.exp(-0.5 * ((xr/sx)**2 + (yr/sy)**2))

    # --- Synthetic B68-like extinction field ---
    data = (
        gauss2d( 10,  20, 16, 14, 3.2, angle= 15) +   # dense core
        gauss2d( 15,  10, 50, 42, 1.5, angle= 20) +   # main body
        gauss2d( 35,  55, 30, 22, 0.5, angle= 35) +   # NW extension
        gauss2d(-10, -20, 38, 28, 0.4, angle= -5) +   # S extension
        gauss2d( 55,  25, 22, 18, 0.3, angle= 10)     # outer structure
    )

    # Add structured noise, smooth, clip
    noise = gaussian_filter(np.random.randn(res, res), sigma=8) * 0.08
    data  = gaussian_filter(np.clip(data + noise, 0, None), sigma=2)

    # --- Scaling ---
    nh2_max  = 3.2                  # max N(H2) / 1e22 cm^-2
    av_scale = 36.0 / nh2_max       # A_V (mag) per N(H2)/1e22

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 7))

    # Filled contours (colour map)
    lev_fill = np.linspace(0, nh2_max, 60)
    cf = ax.contourf(xx, yy, data, levels=lev_fill, cmap='jet', extend='max')

    # Overlaid black isocontour lines
    lev_line = np.linspace(0.15, nh2_max, 18)
    ax.contour(xx, yy, data, levels=lev_line,
            colors='black', linewidths=0.5, alpha=0.6)

    # --- NIRSpec extraction line & slitlet dots ---
    t    = np.linspace(0, 1, 300)
    ra_l  = 60 - 185*t
    dec_l = 70 - 150*t
    ax.plot(ra_l, dec_l, 'w-', lw=1.5, zorder=5)

    t_d = np.linspace(0.05, 0.95, 16)
    ax.plot(60 - 185*t_d, 70 - 150*t_d, 'wo', ms=4, zorder=6)

    # --- Axes formatting ---
    ax.set_xlim( 125, -125)    # RA increases to the left (astronomical convention)
    ax.set_ylim(-130,  130)
    ax.set_xlabel('Right Ascension Offset (arcsec)', fontsize=11)
    ax.set_ylabel('Declination Offset (arcsec)',      fontsize=11)
    ax.set_title('B68 extinction map with extraction positions',
                fontsize=12, pad=8)

    ax.text(0.5, 0.975,
            'Offsets relative to (RA, Dec) = (260.66257, \u221223.83298)',
            transform=ax.transAxes, ha='center', va='top',
            color='white', fontsize=8.5,
            bbox=dict(facecolor='#1f5fa6', edgecolor='none', alpha=0.8, pad=3))

    # --- Dual colorbar: N(H2) left axis, A_V right axis ---
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_ylim(0, nh2_max)
    cbar.set_label(r'$N(\mathrm{H_2})\,/\,10^{22}\ \mathrm{cm}^{-2}$', fontsize=10)
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    ax_av = cbar.ax.twinx()
    ax_av.set_ylim(0, nh2_max * av_scale)   # 0 – 36 mag
    ax_av.set_yticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    ax_av.set_ylabel(r'$A_V\ \mathrm{(mag)}$', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def contour_sample():
    print(x_input.shape)
    r_input = np.linalg.norm(x_input, axis=1)
    #fig, ax = plt.subplots()
    #ax.scatter(x_input[:,0], x_input[:,1], s = 1)
    #ax.hist(r_input, bins=tmplib.__sample_size__//100)
    #plt.show()
    #plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(x_input[:,1], x_input[:,2], s=1, alpha=0.3)

    # Compute KDE
    xy = np.vstack([x_input[:,0], x_input[:,1]])
    kde = gaussian_kde(xy)

    # Build a grid and evaluate the KDE on it
    x_grid = np.linspace(x_input[:,0].min(), x_input[:,0].max(), 200)
    y_grid = np.linspace(x_input[:,1].min(), x_input[:,1].max(), 200)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Overlay isocontours
    ax.contour(xx, yy, zz, levels=10, cmap='viridis')

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    mimic_b68()