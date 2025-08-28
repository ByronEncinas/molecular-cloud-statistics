import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from scipy.spatial import distance
from scipy.spatial import cKDTree

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True

""" Toggle Parameters """

""" Constants and convertion factor """

hydrogen_mass = 1.6735e-24 # gr

# Unit Conversions
km_to_parsec = 1 / 3.085677581e13  # 1 pc in km
pc_to_cm = 3.086 * 10e+18  # cm per parsec
AU_to_cm = 1.496 * 10e+13  # cm per AU
parsec_to_cm3 = 3.086e+18  # cm per parsec
surface_to_column = 2.55e+23

# Physical Constants
mass_unit = 1.99e33  # g
length_unit = 3.086e18  # cm
velocity_unit = 1e5  # cm/s
time_unit = length_unit / velocity_unit  # s
seconds_in_myr = 1e6 * 365.25 * 24 * 3600  # seconds in a million years (Myr)
boltzmann_constant_cgs = 1.380649e-16  # erg/K
grav_constant_cgs = 6.67430e-8  # cm^3/g/s^2
hydrogen_mass = 1.6735e-24  # g

# Code Units Conversion Factors
myrs_to_code_units = seconds_in_myr / time_unit
code_units_to_gr_cm3 = 6.771194847794873e-23  # conversion from code units to g/cm^3
gauss_code_to_gauss_cgs = (1.99e+33/(3.086e+18*100_000.0))**(-1/2)

# ISM Specific Constants
mean_molecular_weight_ism = 2.35  # mean molecular weight of the ISM
gr_cm3_to_nuclei_cm3 = 6.02214076e+23 / (2.35) * 6.771194847794873e-23  # Wilms, 2000 ; Padovani, 2018 ism mean molecular weight is # conversion from g/cm^3 to nuclei/cm^3

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

# cache-ing spatial.cKDTree(Pos[:]).query(x, k=1)
_cached_tree = None
_cached_pos = None

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
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(
        x, Bfield, Density, Density_grad, Pos, VoronoiPos
    )
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1, (3, 1)).T
    CellVol = Volume[cells]
    scaled_dx = 0.5 * dx * ((3/4) * CellVol / np.pi)**(1/3)

    x_tilde = x + scaled_dx[:, np.newaxis] * local_fields_1

    local_fields_2, abs_local_fields_2, _, _ = find_points_and_get_fields(
        x_tilde, Bfield, Density, Density_grad, Pos, VoronoiPos
    )
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2, (3, 1)).T

    x_final = x + 0.5 * scaled_dx[:, np.newaxis] * (local_fields_1 + local_fields_2)

    return x_final, abs_local_fields_1, local_densities, CellVol

def list_files(directory, ext):
    import os
    # List all files in the specified directory
    all_files = os.listdir(directory)
    # Filter out only .npy files
    files = [directory + f for f in all_files if f.endswith(f'{ext}')]
    return files

""" Ionization rate modules """

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

""" Process Lines from File into Lists """

def process_line(line):
    """
    Process a line of data containing information about a trajectory.

    Args:
        line (str): Input line containing comma-separated values.

    Returns:
        dict or None: A dictionary containing processed data if the line is valid, otherwise None.
    """
    parts = line.split(',')
    if len(parts) > 1:
        iteration = int(parts[0])
        traj_distance = float(parts[1])
        initial_position = [float(parts[2:5][0]), float(parts[2:5][1]), float(parts[2:5][2])]
        field_magnitude = float(parts[5])
        field_vector = [float(parts[6:9][0]), float(parts[6:9][1]), float(parts[6:9][2])]
        posit_index = [float(parts[9:][0]), float(parts[9:][1]), float(parts[9:][2])]

        data_dict = {
            'iteration': int(iteration),
            'trajectory (s)': traj_distance,
            'Initial Position (r0)': initial_position,
            'field magnitude': field_magnitude,
            'field vector': field_vector,
            'indexes': posit_index
        }

        return data_dict
    else:
        return None
    
""" Energies """

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

        # Create a mosaic layout with two subplots: one for 'numb', one for 'bfield'
        fig, axs_dict = plt.subplot_mosaic([['numb', 'bfield']], figsize=(12, 5))
        axs_numb = axs_dict['numb']
        axs_bfield = axs_dict['bfield']

        def plot_field(axs, data, label):

            axs.plot(data, label=label)
            if below100 is not None:
                axs.vlines(below100, data[below100]*(1 - 0.1), data[below100]*(1 + 0.1),
                        color='black', label='th 100cm⁻³ (left)')
            if above100 is not None:
                axs.vlines(above100, data[above100]*(1 - 0.1), data[above100]*(1 + 0.1),
                        color='black', label='th 100cm⁻³ (right)')
            if peaks is not None:
                axs.plot(indexes, data[indexes], "x", color="green", label="all peaks")
                axs.plot(indexes, data[indexes], ":", color="green")

            if idx is not None and upline is not None:
                axs.plot(idx, np.max(data), "x", color="black", label="index_global_max")

            axs.axhline(np.min(data), linestyle="--", color="gray", label="baseline")
            axs.set_yscale('log')
            axs.set_xlabel("Index")
            axs.set_ylabel(label)
            axs.set_title(f"{label} Shape")
            axs.legend()
            axs.grid(True)

        # Plot both subplots
        #plot_field(axs_numb, numb, "Density")
        #plot_field(axs_bfield, bfield, "Magnetic Field")

        #plt.tight_layout()
        #plt.savefig('./images/columns/mosaic.png')
        #plt.close(fig)

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

def find_vector_in_array(radius_vector, x_init):
    """
    Finds the indices of the vector x_init in the multidimensional numpy array radius_vector.
    
    Parameters:
    radius_vector (numpy.ndarray): A multidimensional array with vectors at its entries.
    x_init (numpy.ndarray): The vector to find within radius_vector.
    
    Returns:
    list: A list of tuples, each containing the indices where x_init is found in radius_vector.
    """
    x_init = np.array(x_init)
    return np.argwhere(np.all(radius_vector == x_init, axis=-1))

def magnitude(v1, v2=None):
    if v2 is None:
        return np.linalg.norm(v1)  # Magnitude of a single vector
    else:
        return np.linalg.norm(v1 - v2)  # Distance between two vectors

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
