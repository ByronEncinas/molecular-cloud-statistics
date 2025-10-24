import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True

# Set up figure and axes
fig, axs = plt.subplots(2, 3, figsize=(15, 8),gridspec_kw={'wspace': 0, 'hspace': 0},sharex=True, sharey='row')

# Example parameters (e.g., column labels)
column_labels = [r'$\Sigma = 1\,\mathrm{g\,cm^{-2}}$', 
                 r'$\Sigma = 100\,\mathrm{g\,cm^{-2}}$', 
                 r'$\Sigma = 1000\,\mathrm{g\,cm^{-2}}$']

# X-axis data
E = np.logspace(4, 11, 300)
logE = np.log10(E)

# Dummy Y data
def dummy_curve(scale=1, slope=-2):
    return np.log10(scale * E**slope + 1e-30)

# Line styles
line_styles_top = {
    'BS($e^-$)': {'color': 'orange', 'linewidth': 2},
    '$\pi^0$': {'color': 'blue', 'linewidth': 2},
    'BS($e^+$)': {'color': 'green', 'linewidth': 2},
    'total': {'color': 'black', 'linewidth': 2, 'linestyle': '--'}
}

line_styles_bottom = {
    'CR $e^-$': {'color': 'red', 'linewidth': 2},
    '$e^-$ (CR $p$)': {'color': 'orange', 'linewidth': 2, 'linestyle': '--'},
    '$e^-$ (CR $e^-$)': {'color': 'gold', 'linewidth': 2, 'linestyle': ':'},
    '$e^+$ (pair)': {'color': 'green', 'linewidth': 2, 'linestyle': '-.'},
    '$e^+$ ($\pi^+$)': {'color': 'darkgreen', 'linewidth': 2, 'linestyle': '--'},
    '$e^+$ ($\pi^-$)': {'color': 'gray', 'linewidth': 2, 'linestyle': ':'},
    'total': {'color': 'black', 'linewidth': 2}
}

# Loop over columns
for i in range(3):
    # Top row: photon spectra
    ax_top = axs[0, i]
    for label, style in line_styles_top.items():
        ax_top.plot(logE, dummy_curve(scale=10**(-6 - i), slope=-2 + 0.1*i), label=label, **style)
    ax_top.set_title(column_labels[i], fontsize=14)
    if i == 0:
        ax_top.set_ylabel(r'$\log_{10}[j_\gamma / (\mathrm{eV^{-1}\,cm^{-2}\,s^{-1}\,sr^{-1}})]$', fontsize=12)
    ax_top.grid(True)

    # Bottom row: electron/positron spectra
    ax_bottom = axs[1, i]
    for label, style in line_styles_bottom.items():
        ax_bottom.plot(logE, dummy_curve(scale=10**(-8 - i), slope=-2.5 + 0.1*i), label=label, **style)
    if i == 0:
        ax_bottom.set_ylabel(r'$\log_{10}[j_e / (\mathrm{eV^{-1}\,cm^{-2}\,s^{-1}\,sr^{-1}})]$', fontsize=12)
    ax_bottom.set_xlabel(r'$\log_{10}[E/\mathrm{eV}]$', fontsize=12)
    ax_bottom.grid(True)

# Legend handling (only one legend for each row)
axs[0, 2].legend(loc='upper right', fontsize=9)
axs[1, 2].legend(loc='lower left', fontsize=9)

# Adjust layout
plt.tight_layout()
plt.savefig('./six.png')
plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

N = 10_000

# First Gaussian cluster (centered at (0,0))
x1 = np.random.normal(loc=0.0, scale=0.05, size=N)
y1 = np.random.normal(loc=0.0, scale=0.05, size=N)

# Second Gaussian cluster (centered at (1,1))
x2 = np.random.normal(loc=0.15, scale=0.05, size=N)
y2 = np.random.normal(loc=0.2, scale=0.05, size=N)

# Concatenate to make a bi-modal distribution
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])

# Create grid for KDE
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
X, Y = np.meshgrid(np.linspace(xmin, xmax, 1000),
                   np.linspace(ymin, ymax, 1000))
positions = np.vstack([X.ravel(), Y.ravel()])

# KDE estimate
values = np.vstack([x, y])
kernel = gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x, y, s=5, alpha=0.3, label='scatter points')
ax.contour(X, Y, Z, levels=10, colors='k')  # level curves
ax.set_title("KDE with Contours")
plt.legend()
plt.savefig('./kde_curves.png')
plt.close(fig)

# Compute 2D histogram
H, xedges, yedges = np.histogram2d(x, y, bins=100, density=True)
X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                   (yedges[:-1] + yedges[1:]) / 2)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x, y, s=5, alpha=0.3)
contour = ax.contour(X, Y, H.T, levels=10, cmap='viridis')  # Note the .T
ax.clabel(contour, inline=True, fontsize=8)
ax.set_title("2D Histogram with Contours")
plt.savefig('./hst_curves.png')
plt.close(fig)

import matplotlib.pyplot as plt

mask = (y > -0.1) & (y < 0.1)

fig, ax = plt.subplots()
x_masked = x[mask]
b = len(x_masked)//10

ax.hist(x_masked, bins=b)
ax.set_title(r"Slice along $y \approx 0$")
ax.set_xlabel("x")
ax.set_ylabel("Count")
fig.savefig('./hst_slice.png')
plt.close(fig)


# --- Dummy data for illustration ---
x = np.logspace(0, 3.5, 200)   # Σ [g cm^-2]
y1 = -14 - np.log10(x+1)       # example curve
y2 = -15 - np.log10(x+10)
y3 = -16 - np.log10(x+100)

# Inset dummy data
E = np.logspace(5, 13, 200)    # Energy [eV]
J1 = -20 + 5*np.exp(-((np.log10(E)-8)/2)**2)
J2 = J1 - 2
J3 = J1 - 4

# --- Main figure ---
fig, ax = plt.subplots(figsize=(6,6))

# Plot main curves
ax.plot(x, y1, 'k--', label=r"GCRs (Sun av. modul.)")
ax.plot(x, y2, 'k-.', label=r"GCRs (TT min. modul.)")
ax.plot(x, y3, 'k:',  label=r"GCRs (TT max. modul.)")

# Example horizontal line
ax.axhline(-22.8, color='k', linestyle='--', label="LLR")

# Labels
ax.set_xlabel(r"$\Sigma \, [\mathrm{g \, cm^{-2}}]$")
ax.set_ylabel(r"$\log_{10}\,\zeta \, [\mathrm{midplane \, s^{-1}}]$")
ax.set_xscale("linear")  # could be log
ax.set_yscale("linear")

# Top x-axis (secondary axis for N(H2))
def sigma_to_NH2(x):
    return x / (2.3e-24) / 1e26   # example scaling
def NH2_to_sigma(N):
    return N * (2.3e-24) * 1e26

ax_top = ax.secondary_xaxis('top', functions=(sigma_to_NH2, NH2_to_sigma))
ax_top.set_xlabel(r"$N(\mathrm{H}_2) \; [10^{26} \, \mathrm{cm}^{-2}]$")

# Legend
ax.legend(fontsize=9, loc="lower left")

# --- Inset axes ---
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_inset = inset_axes(ax, width="50%", height="50%", loc="upper right")

ax_inset.plot(np.log10(E), J1, 'k--', label="SCRs (active TT)")
ax_inset.plot(np.log10(E), J2, 'k-.', label="GCRs (TT min. modul.)")
ax_inset.plot(np.log10(E), J3, 'k:',  label="GCRs (TT max. modul.)")

ax_inset.set_xlabel(r"$\log_{10} E \, [\mathrm{eV}]$", fontsize=8)
ax_inset.set_ylabel(r"$\log_{10} J_p \, [\mathrm{eV^{-1}\, cm^{-2}\, s^{-1}\, sr^{-1}}]$", fontsize=8)
ax_inset.tick_params(axis='both', which='major', labelsize=8)

# Text annotation inside inset
ax_inset.text(9, -5, "proton spectra\nat 1 AU", fontsize=9)

fig.savefig('./F10.png')
plt.close(fig)

# --- Dummy data for illustration ---
np.random.seed(0)
x_H = -7.5 + 0.2*np.random.randn(2000)   # log10(X(e)) for H
y_H = -16.2 + 0.2*np.random.randn(2000)  # log10(zeta2) for H

x_L = -7.3 + 0.2*np.random.randn(2000)   # log10(X(e)) for L
y_L = -17.0 + 0.2*np.random.randn(2000)  # log10(zeta2) for L

# --- Figure and axes ---
fig, ax = plt.subplots(figsize=(5,4))

# 2D histogram for H region
h_H = ax.hist2d(x_H, y_H, bins=60, cmap="viridis", norm="log")[3]
# 2D histogram for L region
h_L = ax.hist2d(x_L, y_L, bins=60, cmap="magma", norm="log")[3]

# Colorbar (shared scale)
cbar = plt.colorbar(h_H, ax=ax)
cbar.set_label("Pixel count")

# Fit lines (example linear fits)
coeff_H = np.polyfit(x_H, y_H, 1)
coeff_L = np.polyfit(x_L, y_L, 1)

xfit = np.linspace(-8.2, -6.5, 200)
yfit_H = np.polyval(coeff_H, xfit)
yfit_L = np.polyval(coeff_L, xfit)

ax.plot(xfit, yfit_H, "k-", lw=1)       # solid line for H
ax.plot(xfit, yfit_L, "k--", lw=1)      # dashed line for L

# Axis labels
ax.set_xlabel(r"$\log_{10}(X(e^-))$")
ax.set_ylabel(r"$\log_{10}(\zeta_2)\,[\mathrm{s}^{-1}]$")

# Annotations for H, L, and C
ax.text(-7.9, -16.0, r"$\mathcal{H}$", fontsize=18, style="italic")
ax.text(-7.9, -17.2, r"$\mathcal{L}$", fontsize=18, style="italic")
ax.text(-7.5, -16.6, r"$\mathcal{C}$", fontsize=16)

# Example error bar (for "C" region)
ax.errorbar(-7.5, -16.6, xerr=0.1, yerr=0.1, fmt="k.", capsize=4)

# Adjust ticks and limits
ax.set_xlim(-8.2, -6.5)
ax.set_ylim(-17.3, -15.8)

plt.tight_layout()
fig.savefig('./F6.png')
plt.close(fig)



import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Create sample data
X = np.linspace(0, 10, 100)
y = 2.5 * X + 5 + np.random.normal(0, 2, 100)
df = pd.DataFrame({'X': X, 'y': y})

# Fit the linear regression model
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm).fit()

# Get the prediction results
predictions = model.get_prediction(sm.add_constant(df['X']))
summary_frame = predictions.summary_frame(alpha=0.05)

# Extract the bands
pred_band_lower = summary_frame['obs_ci_lower']
pred_band_upper = summary_frame['obs_ci_upper']
conf_band_lower = summary_frame['mean_ci_lower']
conf_band_upper = summary_frame['mean_ci_upper']

# Plot the results
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot the data points
plt.scatter(df['X'], df['y'], s=20, color='darkblue', alpha=0.7, label='Data Points')

# Plot the regression line
plt.plot(df['X'], model.fittedvalues, color='red', linewidth=2.5, label='Regression Line')

# Plot the prediction band
plt.fill_between(df['X'], pred_band_lower, pred_band_upper, color='orange', alpha=0.3, label='95% Prediction Band')

# Plot the confidence band
plt.fill_between(df['X'], conf_band_lower, conf_band_upper, color='green', alpha=0.5, label='95% Confidence Band')

plt.title('95% Confidence and Prediction Bands', fontsize=18)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.legend(fontsize=12, loc='upper left')

# Adjust plot aesthetics
plt.margins(0.05)
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig('95_percent_bands_plot.png')