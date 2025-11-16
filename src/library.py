import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.spatial import cKDTree
from scipy import interpolate
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False

""" Toggle Parameters """

""" Statistics Methods """

def reduction_to_density(factor, numb):
    factor = np.array(factor)
    numb = np.array(numb)
    # R is numpy array
    def match_ref(n, d_data, r_data, p_data=0):
        sample_r = []

        for i in range(0, len(d_data)):
            if np.abs(np.log10(d_data[i]/n)) < 1/8:
                sample_r.append(r_data[i])
        sample_r.sort()
        try:
            mean = sum(sample_r)/len(sample_r)
            median = np.quantile(sample_r, .5)
            ten = np.quantile(sample_r, .1)
            size = len(sample_r)
        except:
            return [sample_r*0, np.nan, np.nan, np.nan, 0]
        return [sample_r, mean, median, ten, size]

    total = factor.shape[0]
    #numb = numb[factor<1]
    #factor = factor[factor<1]    
    x_n = np.logspace(np.min(np.log10(numb)), np.max(np.log10(numb)), total)
    mean_vec = np.zeros(total)
    median_vec = np.zeros(total)
    ten_vec = np.zeros(total)
    sample_size = np.zeros(total)
    matrix  = []
    for i in range(0, total):
        s = match_ref(x_n[i], numb, factor)
        matrix+=[[s[0]]]
        mean_vec[i] = s[1]
        median_vec[i] = s[2]
        ten_vec[i] = s[3]
        sample_size[i] = s[4]
    
    return x_n, matrix, mean_vec, median_vec, ten_vec, sample_size

def get_globals_memory() -> None:
    import sys

    total = 0
    for name, obj in globals().items():
        if name.startswith("__") and name.endswith("__"):
            continue  # skip built-in entries
        try:
            total += sys.getsizeof(obj)
        except TypeError:
            pass  # some objects might not report size

    # Convert bytes → gigabytes
    gb = total / (1024 ** 3)
    print(f"Memory used by globals: {gb:.6f} gigabytes")

""" Constants and convertion factor """

# Unit Conversions
gauss_to_micro_gauss = 1e+6
km_to_parsec = 1 / 3.085677581e13  # 1 pc in km
pc_to_cm = 3.086 * 1.0e+18  # cm per parsec
AU_to_cm = 1.496 * 1.0e+13  # cm per AU
surface_to_column = 2.55e+23
pc_myrs_to_km_s = 0.9785

# Physical Constants
mass_unit = 1.99e33  # g
length_unit = 3.086e18  # cm in a parsec
velocity_unit = 1e5  # cm/s
time_unit = length_unit / velocity_unit  # s
seconds_in_myr = 1e6 * 365.25 * 24 * 3600  # seconds in a million years (Myr)
boltzmann_constant_cgs = 1.380649e-16  # erg/K
grav_constant_cgs = 6.67430e-8  # cm^3/g/s^2
hydrogen_mass = 1.6735e-24  # g

# Code Units Conversion Factors
myrs_to_code_units = seconds_in_myr / time_unit
code_units_to_gr_cm3 = 6.771194847794873e-23  # conversion from code units to g/cm^3
gauss_code_to_gauss_cgs = (4 * np.pi)**0.5   * (3.086e18)**(-1.5) * (1.99e33)**0.5 * 1e5 # cgs units

# ISM Specific Constants
mean_molecular_weight_ism = 2.35  # mean molecular weight of the ISM (Wilms, 2000)
gr_cm3_to_nuclei_cm3 = 6.02214076e+23 / (2.35) * 6.771194847794873e-23  # Wilms, 2000 ; Padovani, 2018 ism mean molecular weight is # conversion from g/cm^3 to nuclei/cm^3

# cache-ing spatial.cKDTree(Pos[:]).query(x, k=1)
_cached_tree = None
_cached_pos = None

""" Ionization Rate Parameters"""

""" Arepo Process Methods (written by A. Mayer at MPA July 2024)

(Original Functions Made By A. Mayer (Max Planck Institute) + contributions B. E. Velazquez (University of Texas))
"""
def get_magnetic_field_at_points(x, Bfield, rel_pos):
	n = len(rel_pos[:,0])
	local_fields = np.zeros((n,3))
	for  i in range(n):
		local_fields[i,:] = Bfield[i,:]
	return local_fields

def get_density_at_points(x, Density, Density_grad, rel_pos):
	n = len(rel_pos[:,0])	
	local_densities = np.zeros(n)
	for  i in range(n):
		local_densities[i] = Density[i] + np.dot(Density_grad[i,:], rel_pos[i,:])
	return local_densities

def find_points_and_relative_positions(x, Pos, VoronoiPos):
    dist, cells = spatial.KDTree(Pos[:]).query(x, k=1, workers=-1)
    rel_pos = VoronoiPos[cells] - x
    return dist, cells, rel_pos

def find_points_and_relative_positions_tree_dependent(tree, x, Pos, VoronoiPos):
    dist, cells = _cached_tree.query(x, k=1, workers=-1)
    rel_pos = VoronoiPos[cells] - x
    return dist, cells, rel_pos

def find_points_and_relative_positions(x, Pos, VoronoiPos):
    global _cached_tree, _cached_pos
    if _cached_tree is None or not np.array_equal(Pos, _cached_pos):
        _cached_tree = cKDTree(Pos)
        _cached_pos = Pos.copy()

    dist, cells = _cached_tree.query(x, k=1, workers=-1)
    rel_pos = VoronoiPos[cells] - x
    return dist, cells, rel_pos

def find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos, VoronoiPos)
	local_fields = get_magnetic_field_at_points(x, Bfield[cells], rel_pos)
	local_densities = get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos)
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	return local_fields, abs_local_fields, local_densities, cells
	
def Heun_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume, bdirection=None):

    # campo en x, mangitud campo en x, densidad en x y ID de la celda
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(
        x, Bfield, Density, Density_grad, Pos, VoronoiPos
    )

    # vector unitario en la dirección del campo en x
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1, (3, 1)).T

    # Volume de la celda en la que está x
    CellVol = Volume[cells]

    # radio de esfera con volumen CellVol
    scaled_dx = dx * ((3/4) * CellVol / np.pi)**(1/3)

    # paso intermedio en x
    x_tilde = x + 0.5 * scaled_dx[:, np.newaxis] * local_fields_1

    # campo en x_intermedio, mangitud campo en x_intermedio, densidad en x_intermedio y ID de la celda intermedia
    local_fields_2, abs_local_fields_2, _, _ = find_points_and_get_fields(
        x_tilde, Bfield, Density, Density_grad, Pos, VoronoiPos
    )
    # vector unitario en la dirección del campo en x intermedio
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2, (3, 1)).T

    # promedio entre paso inicial e intermedio
    x_final = x + 0.5 * scaled_dx[:, np.newaxis] * (local_fields_1 + local_fields_2)

    return x_final, abs_local_fields_1, local_densities, CellVol

def Euler_step(x, dx, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume, bdirection=None):

    # campo en x, mangitud campo en x, densidad en x y ID de la celda
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(
        x, Bfield, Density, Density_grad, Pos, VoronoiPos
    )

    # vector unitario en la dirección del campo en x
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1, (3, 1)).T

    # Volume de la celda en la que está x
    CellVol = Volume[cells]

    # radio de esfera con volumen CellVol
    scaled_dx = dx * ((3/4) * CellVol / np.pi)**(1/3)

    # paso final
    x_final = x + 0.5 * scaled_dx[:, np.newaxis] * (local_fields_1)

    return x_final, abs_local_fields_1, local_densities, CellVol

""" Ionization rate modules & constants """

size = 10_000
Ei = 1.0e+3
Ef = 1.0e+9
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

model = None

def select_species(m):

    if m == 'L':
        C = 2.43e+15 *4*np.pi
        alpha, beta = 0.1, 2.8
        Enot = 650e+6
    elif m == 'H':
        C = 2.43e+15 *4*np.pi
        alpha, beta = -0.8, 1.9
        Enot = 650e+6
    elif m == 'e':
        C = 2.1e+18*4*np.pi
        alpha, beta = -1.5, 1.7
        Enot = 710e+6
    else:
        raise ValueError(f"[Error] Argument {m} not supported")
    return C, alpha, beta, Enot

# only for protons
cross_data = np.load('arepo_data/cross_pH2_rel_1e18.npz')
loss_data  = np.load('arepo_data/Kedron_pLoss.npz')

cross = interpolate.interp1d( cross_data["E"], cross_data["sigmap"])
loss = interpolate.interp1d(loss_data["E"], loss_data["L_full"])

def ionization_rate_fit(Neff):
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
    return log_L(Neff), log_H(Neff)

def column_density(radius_vector, magnetic_field, numb_density, direction='', mu_ism = np.logspace(-2, 0, num=50)):
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

    return Nmu, mu_local, dmu, trajectory

def mirrored_column_density(radius_vector, magnetic_field, numb_density, Nmu, direction='', mu_ism = np.logspace(-2, 0, num=50)):

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

    return Nmir

def ionization_rate(Nmu, mu_local, dmu, direction = '',mu_ism = np.logspace(-2, 0, num=50), m='L'):

    C, alpha, beta, Enot = select_species(m)
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))
    loss_function = lambda z: Lstar*(Estar/z)**d
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
            
            jspectra[j,:] = spectra
            #jl_dE = np.sum(isms*llei*diff_energy)  # j_i(E_i)L(E_i) = 10^{log_10(j_i(E_i)) + log_10(L(E_i))}
            #zeta_mui[j, l] = jl_dE / epsilon 
            zeta_mui[j, l] = np.sum(spectra*sigma_ion*diff_energy)

    zeta = np.sum(dmu * zeta_mui, axis = 1)

    return zeta, zeta_mui, jspectra

def local_spectra(Nmu, mu_local, mu_ism = np.logspace(-2, 0, num=50), m='L'):
    C, alpha, beta, Enot = select_species(m)
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))
    loss_function = lambda z: Lstar*(Estar/z)**d 
    jspectra = np.zeros((Nmu.shape[0], energy.shape[0]))

    for l, mui in enumerate(mu_ism):
        for j, Nj in enumerate(Nmu[:,l]):
            mu_ = mu_local[j,l]
            if mu_ <= 0:
                break

            Ei = ((energy)**(1 + d) + (1 + d) * Lstar* Estar**(d)* Nj/mu_)**(1 / (1 + d))

            isms = ism_spectrum(Ei)                        # log_10(j_i(E_i))
            llei = loss_function(Ei)                       # log_10(L(E_i))
            jspectra[j,:] = 0.5*isms*llei/loss_function(energy)  

    return np.sum(jspectra, axis = 0)

def x_ionization_rate(fields, densities, vectors, x_input, m='L'):

    C, alpha, beta, Enot = select_species(m)
    ism_spectrum = lambda x: C*(x**alpha/((Enot + x)**beta))
    loss_function = lambda z: Lstar*(Estar/z)**d

    lines = x_input.shape[0]
    zeta_at_x = np.zeros(lines)
    nmir_at_x = np.zeros(lines)

    local_spectra_at_x = np.zeros((lines, size))

    for line in range(lines):
        density    = densities[:, line]
        field =  fields[:, line]*1e6 # microgauss
        vector  =  vectors[:, line, :]

        # slice out zeroes        
        mask = np.where(density > 0.0)[0]
        start, end = mask[0], mask[-1]

        density    = density[start:end]
        field =  field[start:end] #np.ones_like(field[start:end]) 
        vector  =  vector[start:end, :]
        
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
        
        zeta_mir_fwd, zeta_mui_mir_fwd, spectra_fwd  = ionization_rate(Nmir_fwd, mu_local_fwd, dmu_fwd, 'mir_fwd', m=m)
        zeta_mir_bwd, zeta_mui_mir_bwd, spectra_bwd  = ionization_rate(Nmir_bwd, mu_local_bwd, dmu_bwd, 'mir_bwd', m=m)

        Nmir = np.sum(Nmir_fwd + Nmir_bwd[::-1], axis=1) # Adding the corresponding 
        zeta = (zeta_mir_fwd+ zeta_mir_bwd[::-1])          

        zeta_at_x[line] = zeta[arg_input]
        nmir_at_x[line] = Nmir[arg_input]
        local_spectra_at_x[line, :] = spectra_fwd[arg_input,:] + spectra_bwd[::-1,:][arg_input,:]

        return nmir_at_x, zeta_at_x, local_spectra_at_x
    
""" Energies """

def eval_reduction(field, numb, follow_index, threshold):

    R10      = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    flag = False
    filter_mask = np.ones(m).astype(bool)
    dead = 0
    for i in range(m):

        mask10 = np.where(numb[:, i] > threshold)[0]
        if mask10.size > 0:
            start, end = mask10[0], mask10[-1]
            if start <= follow_index <= end:
                try:
                    numb10   = numb[start:end+1, i]
                    bfield10 = field[start:end+1, i]
                    p_r = follow_index - start
                    B_r = bfield10[p_r]
                    n_r = numb10[p_r]
                except IndexError:
                    raise ValueError(f"\nTrying to slice beyond bounds for column {i}. "
                                    f"start={start}, end={end}, shape={numb.shape}")
            else:
                print(f"\n[Info] follow_index {follow_index} outside threshold interval for column {i}.")
                if follow_index >= numb.shape[0]:
                    raise ValueError(f"follow_index {follow_index} is out of bounds for shape {numb.shape}")
                numb10   = np.array([numb[follow_index, i]])
                bfield10 = np.array([field[follow_index, i]])
                p_r = 0
                B_r = bfield10[p_r]
                n_r = numb10[p_r]
        else:
            print(f"\n[Info] No densities > {threshold} cm-3 found for column {i}. Using follow_index fallback.")
            if follow_index >= numb.shape[0]:
                raise ValueError(f"\nfollow_index {follow_index} is out of bounds for shape {numb.shape}")
            numb10   = np.array([numb[follow_index, i]])
            bfield10 = np.array([field[follow_index, i]])
            p_r = 0
            B_r = bfield10[p_r]
            n_r = numb10[p_r]

        #print("p_r: ", p_r)
        if not (0 <= p_r < bfield10.shape[0]):
            raise IndexError(f"\np_r={p_r} is out of bounds for bfield10 of length {len(bfield10)}")

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield10, numb10, p_r, plot=flag)
        index_pocket, field_pocket = pocket[0], pocket[1]
        flag = False
        p_i = np.searchsorted(index_pocket, p_r)
        from collections import Counter
        most_common_value, count = Counter(bfield10.ravel()) .most_common(1)[0]
    
        if count > 20:
            R = 1.
            #print(f"Most common value: {most_common_value} (appears {count} times): R = ", R)
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)   
            flag = True
            filter_mask[i] = False
            dead +=1
            continue     

        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            B_h = max([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            # YES! 
            success = True  
        except:
            # NO :c
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)
            else:
                # NO!
                R = 1.
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)

    filter_mask = filter_mask.astype(bool)

    return np.array(R10), np.array(Numb100), np.array(B100), filter_mask

def fibonacci_sphere(samples=20):
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    y = np.linspace(1 - 1/samples, -1 + 1/samples, samples)  # Even spacing in y
    radius = np.sqrt(1 - y**2)  # Compute radius for each point
    theta = phi * np.arange(samples)  # Angle increment

    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.vstack((x, y, z)).T  # Stack into a (N, 3) array

def pocket_finder(bfield, numb, p_r, plot=False):
    #pocket_finder(bfield, p_r, B_r, img=i, plot=False)
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    try:
        idx = index_global_max[0]
    except:
        idx = index_global_max
    upline == bfield[idx]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001


    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]

    if plot:
        # Find threshold crossing points for 100 cm^-3
        mask = np.log10(numb) < 2  # log10(100) = 2
        slicebelow = mask[:p_r]
        sliceabove = mask[p_r:]
        peaks = np.array(peaks)
        indexes = np.array(indexes)

        try:
            above100 = np.where(sliceabove)[0][0] + p_r
        except IndexError:
            above100 = None

        try:
            below100 = np.where(slicebelow)[0][-1]
        except IndexError:
            below100 = None


    return (indexes, peaks), (index_global_max, upline)

def smooth_pocket_finder(bfield, cycle=0, plot=False):
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    upline == bfield[index_global_max]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001

    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]
    
    if plot:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.plot(bfield)
        axs.plot(indexes, peaks, "x", color="green")
        axs.plot(indexes, peaks, ":", color="green")
        
        axs.plot(np.ones_like(bfield) * baseline, "--", color="gray")
        axs.set_xlabel("Index")
        axs.set_ylabel("Field")
        axs.set_title("Actual Field Shape")
        axs.legend(["bfield", "all peaks", "index_global_max", "baseline"])
        axs.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"./field_shape{cycle}.png")
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

def find_insertion_point(array, val):

    for i in range(len(array)):
        if val < array[i]:
            return i  # Insert before index i
    return len(array)  # Insert at the end if p_r is greater than or equal to all elements

def tda(X, distro):
    # persistence diagrams and barcodes to identify structures
    # mapper to transform dataset into a easily understandable graph
    # others
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "The package 'plotly' is not installed. "
            "Install it with:\n\n    pip install plotly"
        )

    try:
        from ripser import ripser
    except ImportError:
        raise ImportError(
            "The package 'ripser' is not installed. "
            "Install it with:\n\n    pip install ripser"
        )

    try:
        from persim import plot_diagrams
    except ImportError:
        raise ImportError(
            "The package 'persim' is not installed. "
            "Install it with:\n\n    pip install persim"
        )

    diagrams = ripser(X, maxdim=2)["dgms"]

    # Plot the diagram
    fig = plt.figure()
    plot_diagrams(diagrams)

    # Save it to a file (e.g., PNG or PDF)
    plt.savefig(f"images/xyz_distro/pd_{distro}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def dendogram_analysis(DensityField):

    from astrodendro import Dendrogram
    d = Dendrogram.compute(DensityField, min_value=10e+4)
    p = d.plotter()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot the whole tree
    p.plot_tree(ax, color='black')

    # Highlight two branches
    p.plot_tree(ax, structure=8, color='red', lw=2, alpha=0.5)
    p.plot_tree(ax, structure=24, color='orange', lw=2, alpha=0.5)

    # Add axis labels
    ax.set_xlabel("Structure")
    ax.set_ylabel("Flux")

def use_lock_and_save(path):
    from filelock import FileLock
    """
    Time ──────────────────────────────▶

    Script 1:  ──[Acquire Lock]─────[Create/Append HDF5]─────[Release Lock]───

    Script 2:             ──[Wait for Lock]─────────────[Append HDF5]─────[Release Lock]───
    """
    lock = FileLock(path + ".lock")

    with lock:  # acquire lock (waits if another process is holding it)

        pass
        # Use this to create and update a dataframe

""" Decorators 
Obtained from: https://towardsdatascience.com/python-decorators-for-data-science-6913f717669a/

I like this for runs that take time and are not in parallel
"""

def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

import smtplib
import traceback
from email.mime.text import MIMEText

def email_on_failure(sender_email, password, recipient_email):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # format the error message and traceback
                err_msg = f"Error: {str(e)}nnTraceback:n{traceback.format_exc()}"

                # create the email message
                message = MIMEText(err_msg)
                message['Subject'] = f"{func.__name__} failed"
                message['From'] = sender_email
                message['To'] = recipient_email

                # send the email
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(sender_email, password)
                    smtp.sendmail(sender_email, recipient_email, message.as_string())
                # re-raise the exception
                raise
        return wrapper
    return decorator

import time
from functools import wraps

def retry(max_tries=3, delay_seconds=1):
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        raise e
                    time.sleep(delay_seconds)
        return wrapper_retry
    return decorator_retry