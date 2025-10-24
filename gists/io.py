import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib.ticker import MaxNLocator
import csv
import os

Ei = 1.0e+0
Ef = 1.0e+15

N0 = 10e+19
Nf = 10e+27

n0 = 150 # cm−3
k  = 0.5 # Spectral index

d = 0.82
a = 0.1 # Spectral index either 0.1 from Low Model, or \in [0.5, 2.0] according to free streaming analytical solution.

# Mean energy lost per ionization event
epsilon = 35.14620437477293

# Luminosity per unit volume for cosmic rays (eV cm^2)
Lstar = 1.4e-14

# Flux constant (eV^-1 cm^-2 s^-1 sr^-1) C*(10e+6)**(0.1)/(Enot+6**2.8)
Jstar = 2.43e+15*(10e+6)**(0.1)/(500e+6**2.8) # Proton in Low Regime (A. Ivlev 2015)

# Energy scale for cosmic rays (1 MeV = 1e+6 eV)
Estar = 1.0e+6

# Column densities
column_density = np.logspace(19, 27, num=10)

# Model H coefficients
model_H = [1.001098610761e7, -4.231294690194e6, 7.921914432011e5,
           -8.623677095423e4, 6.015889127529e3, -2.789238383353e2,
           8.595814402406e0, -1.698029737474e-1, 1.951179287567e-3,
           -9.937499546711e-6]

# Model L coefficients
model_L = [-3.331056497233e+6, 1.207744586503e+6, -1.913914106234e+5,
           1.731822350618e+4, -9.790557206178e+2, 3.543830893824e+1,
           -8.034869454520e-1, 1.048808593086e-2, -6.188760100997e-5,
           3.122820990797e-8]

# Calculate log(zeta) for Model L
logzl = []
for i, Ni in enumerate(column_density):
    lzl = sum([cj * (np.log10(Ni))**j for j, cj in enumerate(model_L)])
    logzl.append(lzl)

print("Extremes of fitting zeta(N) for Model L: ", logzl[0], logzl[-1])

logzetalfit = np.array(logzl)

# Calculate log(zeta) for Model H
logzh = []
for i, Ni in enumerate(column_density):
    lzh = sum([cj * (np.log10(Ni))**j for j, cj in enumerate(model_H)])
    logzh.append(lzh)

print("Extremes of fitting zeta(N) for Model H: ", logzh[0], logzh[-1])

logzetahfit = np.array(logzh)

svnteen = np.ones_like(column_density) * (-17)

# Plotting

# Free Streaming Cosmic Rays, Analytical Expression
fig, ax = plt.subplots()  # Use a single plot

If = {
    "a=0.4": 5.14672,
    "a=0.5": 3.71169,
    "a=0.7": 2.48371,
    "a=0.9": 1.92685,
    "a=1.1": 1.60495
}

Nof = Estar / ((1 + d) * Lstar)

# Colors for plotting
colors = cm.rainbow(np.linspace(0, 1, len(If)))

# Plot each curve and add text annotations
for c, (b, I) in zip(colors, If.items()):
    free_streaming_ion = []
    a = float(b.split("=")[1])
    gammaf = (a + d - 1) / (1 + d)
    
    for Ni in column_density:
        fs_ionization = 4 * np.pi * (1 / epsilon) * (1 + d) / (a + 2 * d) * Jstar * Lstar * Estar * I * (Ni / Nof) ** (-gammaf)
        free_streaming_ion.append(np.log10(fs_ionization))
    
    ax.plot(column_density, free_streaming_ion, label=f'$log(\\zeta_f(N, a={a}))$', linestyle="--", color=c)
    ax.text(column_density[-1], free_streaming_ion[-1], f'a={a}, $\\gamma_f$={gammaf:.2f}', fontsize=10, color=c)

# Plot Ionization vs Column Density (First plot)
ax.plot(column_density, logzetalfit, label='$\\mathcal{L}$', linestyle="--", color="grey")
ax.plot(column_density, logzetahfit, label='$\\mathcal{H}$', linestyle="--", color="dimgrey")
ax.plot(column_density, svnteen, label='$\\zeta = 10^{-17}$', linestyle=":", color="dimgrey")
ax.set_xscale('log')
ax.set_title('Ionization vs Column Density')
ax.set_xlabel('$N cm^{-2}$ (log scale)')
ax.set_ylabel('$\\zeta(N)$ (log scale)')
ax.legend()
ax.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig("./ionizations.png")
plt.show()


import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# Model L coefficients (same as before)
model_L = [-3.331056497233e+6,  1.207744586503e+6,-1.913914106234e+5,
            1.731822350618e+4, -9.790557206178e+2, 3.543830893824e+1, 
           -8.034869454520e-1,  1.048808593086e-2,-6.188760100997e-5, 
            3.122820990797e-8]

# Column density range
column_density = np.logspace(19, 27, num=10)

# Interpolation function (using scipy's interp1d)
logzetalfit_interpolate = interp.interp1d(np.log10(column_density), logzl, kind='linear', fill_value="extrapolate")

# Evaluate the function at new column densities (including in between values)
new_column_density = np.logspace(19, 27, num=50)  # 50 points for higher resolution
logzl_interpolated = logzetalfit_interpolate(np.log10(new_column_density))

# Plot original log(zeta) and interpolated values
plt.figure(figsize=(10, 6))
plt.plot(column_density, logzl, label='Original log(zeta)', linestyle="--", color="blue")
plt.plot(new_column_density, logzl_interpolated, label='Interpolated log(zeta)', color="red")
plt.xscale('log')
plt.xlabel('Column Density $N_{cm^{-2}}$')
plt.ylabel('log(zeta(N))')
plt.title('log(zeta(N)) vs Column Density')
plt.legend()
plt.grid(True)
plt.show()

# before collapse (ideal)
print(logzetalfit_interpolate(22) / logzetalfit_interpolate(23))
# after collapse (ideal)
print(logzetalfit_interpolate(22.5) / logzetalfit_interpolate(23.5))

# before collapse (non-ideal)
print(logzetalfit_interpolate(22) / logzetalfit_interpolate(23))
# after collapse (non-ideal)
print(logzetalfit_interpolate(22.5) / logzetalfit_interpolate(23.5))
