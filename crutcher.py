import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict
import numpy as np
import sys, time, glob, re, os
import warnings, csv
import random


#...............Core Density Comparison..................


cases = ['ideal', 'amb']

s_ideal = []
s_amb = []
t_ideal = []
t_amb = []
nc_ideal = []
nc_amb = []

for case in cases:
    file_path = f'./{case}_cloud_trajectory.txt'
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if case == "ideal":
                s_ideal.append(str(row[0]))
                t_ideal.append(float(row[1]))
                nc_ideal.append(float(row[-1]))

            if case == "amb":
                s_amb.append(str(row[0]))
                t_amb.append(float(row[1]))
                nc_amb.append(float(row[-1]))

mu_mH = 2.35 * 1.67e-24
rho_ideal = np.array(nc_ideal) * mu_mH
rho_amb = np.array(nc_amb) * mu_mH

def evaluate_reduction(field, numb, thresh):
    R10 = []
    R100 = []
    Numb100 = []
    RBundle = []
    m = field.shape[1]

    for i in range(m):

        shape = magnetic_fields.shape[0]

        # threshold, threshold2, threshold_rev, threshold2_rev
        x10, x100, y10, y100 = threshold[:,i]

        # to slice bfield with threshold 10cm-3
        xp10 = shape//2 + x10
        xm10 = shape//2 - y10

        try:
            numb   = numb_densities[xm10-1:xp10+1,i]
            bfield = magnetic_fields[xm10-1:xp10+1,i]
        except:
            raise ValueError("Trying to slice a list outisde of its boundaries")

        p_r = shape//2 - xm10-1
        B_r = bfield[p_r]
        n_r = numb[p_r]

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield, p_r, B_r, img=i, plot=False)
        index_pocket, field_pocket = pocket[0], pocket[1]

        p_i = np.searchsorted(index_pocket, p_r)
        
        # are there local maxima around our point? 
        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
            B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
            # YES! 
            success = True  
        except:
            # NO :c
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R10.append(R)
                Numb100.append(n_r)
            else:
                # NO!
                R = 1.
                R10.append(R)
                Numb100.append(n_r)
        del closest_values, success, B_l, B_h, R

        # pocket with density threshold 100cm-3
        mask = np.log10(numb)<2
        slicebelow = mask[:p_r]
        sliceabove = mask[p_r:]

        above100 = np.where(sliceabove == True)[0][0] + p_r
        below100 = np.where(slicebelow == True)[0][-1]    

        numb   = numb[below100-1:above100+1]
        bfield = bfield[below100-1:above100+1]

        # original size N, new size N'
        # cuts where done from 0 - below100 and above100 - N
        # p_r is the index of the generating point
        # what is p_r?
        
        p_r = p_r - below100 + 1

        #print(p_r, np.round(bfield[p_r], 4))
        
        B_r = bfield[p_r]

        pocket, global_info = pocket_finder(bfield, p_r, B_r, plot=False)
        index_pocket, field_pocket = pocket[0], pocket[1]

        p_i = np.searchsorted(index_pocket, p_r)

        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield[closest_values[0]], bfield[closest_values[1]]])
            B_h = max([bfield[closest_values[0]], bfield[closest_values[1]]])
            success = True  
        except:
            R = 1
            R100.append(R)

            success = False 
            continue
        if success:
            if B_r / B_l < 1:
                R = 1 - np.sqrt(1 - B_r / B_l)
                R100.append(R)

            else:
                R = 1
                R100.append(R)

        #print(R10[-1] - R100[-1])
        RBundle.append((R10, R100))

    return RBundle, R10, R100, Numb100

def statistics_reduction(R, N):
    R = np.array(R)
    N = np.array(N)
    # R is numpy array
    def _stats(n, d_data, r_data, p_data=0):
        sample_r = []

        for i in range(0, len(d_data)):
            if np.abs(np.log10(d_data[i]/n)) < 1/8:
                sample_r.append(r_data[i])
        sample_r.sort()
        if len(sample_r) == 0:
            mean = None
            median = None
            ten = None
            size = 0
        else:
            mean = sum(sample_r)/len(sample_r)
            median = np.quantile(sample_r, .5)
            ten = np.quantile(sample_r, .1)
            size = len(sample_r)
        return [mean, median, ten, size]

    total = len(R)
    ones  = np.sum(R==1)
    not_ones  = total - ones
    #print(ones, total)
    f = ones/total
    ncrit = 100
    mask = R<1
    R = R[mask]
    N = N[mask]
    minimum, maximum = np.min(np.log10(N)), np.max(np.log10(N))
    Npoints = len(R)

    x_n = np.logspace(minimum, maximum, Npoints)
    mean_vec = np.zeros(Npoints)
    median_vec = np.zeros(Npoints)
    ten_vec = np.zeros(Npoints)
    sample_size = np.zeros(Npoints)
    for i in range(0, Npoints):
        s = _stats(x_n[i], N, R)
        mean_vec[i] = s[0]
        median_vec[i] = s[1]
        ten_vec[i] = s[2]
        sample_size[i] = s[3]
    
    num_bins = Npoints//10  # Define the number of bins as a variable

    return R, x_n, mean_vec, median_vec, ten_vec, sample_size, f, N

pc_to_cm = 3.086 * 10e+18  # cm per parsec

start_time = time.time()

amb_bundle   = sorted(glob.glob('./thesis_stats/amb/*/DataBundle*.npz'))
ideal_bundle = sorted(glob.glob('./thesis_stats/ideal/*/DataBundle*.npz'))

bundle_dirs = [ideal_bundle,amb_bundle]

for bundle_dir in bundle_dirs: # ideal and ambipolar
    if bundle_dir == []:
        continue
    case = str(bundle_dir[0].split('/')[-3])
    repeated = set()
    for snap_data in bundle_dir: # from 000 to 490 
        snap = str(snap_data.split('/')[-2])
        data = np.load(snap_data, mmap_mode='r')
        numb_densities = data['number_densities']
        magnetic_fields = data['magnetic_fields']

fig, ax = plt.subplots()
ax.set_ylabel(r'$B_{path}$ ($\mu G$)', fontsize=12)
ax.set_xlabel(r'''$n_H$ ($\rm{cm}^{-3}$)
              
        Magnetic strength to density relation in simulations (blue dots) in 
        comparison with Zeeman measurements of line of sight (solid red) as 
        presented by Crutcher et al (2012)''')

ax.set_yscale('log')
ax.set_xscale('log')

def B_los_fit(n_H, B0=10, n_crit=300, kappa=0.65):
    """
    Parametrize the Clutcher 2012 blue fit line.
    300cm-3 for molecular clouds
    """
    B = np.where(n_H < n_crit, B0, B0 * (n_H / n_crit)**kappa)
    return B

n_H = np.logspace(1, 7, 500)
B = B_los_fit(n_H)

yp = np.log10(magnetic_fields[numb_densities>0]).flatten()
xp = np.log10(numb_densities[numb_densities>0]).flatten()

from scipy import stats

mp, bp, *_ = stats.linregress(xp, yp)

y = 10**(mp*xp+ bp)
x = 10**xp
#ax.plot(n1, b1, color='red')
#ax.plot(n2, b2, color='red')
ax.plot(x,y, color='orange', label='Sim Exp Fit')
ax.plot(n_H, B, color='red', label='Crutcher et al, 2012')
ax.scatter(numb_densities, magnetic_fields, color='dodgerblue', s = 0.5)
ax.tick_params(axis='y')
plt.title(r"$B_{path} \propto n_{g}$")
fig.tight_layout()
plt.grid()
plt.legend()
plt.savefig('./images/crutcher_et_al.png')
plt.close()

"""
Mass to Flux Ratio 
Crutcher (1999) summarized the state of Zeeman observations in molecular clouds and dis-
cussed the results. At that time there were 27 sensitive Zeeman measurements and 15 detections.
That analysis found the following results for the molecular clouds studied. 

(a) Internal motions are supersonic but approximately equal to the Alfven speed. 
(b) The ratio of thermal to magneticpressures is \beta_\rho ≈ 0.04, implying that magnetic fields are 
important in the physics of molecular clouds. 
(c) The mass-to-magnetic flux ratio $M/\phi$ is about twice critical, which suggests that static magnetic 
fields alone are insufficient to support clouds against gravity. 
(d ) Kinetic and magnetic energies are approximately equal, which suggests that turbulent and larger-scale 
regular magnetic fields are roughly equally important in cloud energetics. 
(e) Magnetic field strengths scale with gas densities as BTOT ∝ \rho \kappa with \kappa ≈ 0.47; this agrees 
with the prediction of ambipolar-diffusion-driven star formation. 

"""

"""

\lambda = (M/\Phi)/(M/\Phi)_crit
                =    5.0e-21 N(H)   /B (A. Maury & B. Commercon - Les Houches 2017) or
                =    7.6e-21 N(H_2) /B (Crutcher 2012)

Magnetically supercritical (\lambda > 1)
- E_{grav} > E_{mag}
- Self-Gravitating (collapse)

Magnetically subcritical (\lambda < 1)
- E_{grav} > E_{mag}
- B supports the cloud ()
"""