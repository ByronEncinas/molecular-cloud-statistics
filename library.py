import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import h5py
from scipy import spatial
from scipy.spatial import distance
import sys
import matplotlib
import healpy as hp

""" Arepo Process Methods (written by A. Mayer at MPA July 2024) """


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

""" b_ionization_model Methods """
""" c_reduction_factor Methods """

def pocket_finder(bfield, cycle=0, plot=False):
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
    idx = index_global_max[0]
    upline == bfield[idx]
    scope = index_global_max[-1] - index_global_max[0]

    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield[0:idx+scope]):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield[idx-scope:])):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks + list(reversed(rpeaks))
    indexes = lindex + list(reversed(rindex))
    
    if plot:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

        axs.plot(bfield)
        axs.plot(indexes, peaks, "x", color="green")
        axs.plot(indexes, peaks, ":", color="green")
        
        #for idx in index_global_max:
        axs.plot(idx, upline, "x", color="black")
        axs.plot(np.ones_like(bfield) * baseline, "--", color="gray")
        axs.set_xlabel("Index")
        axs.set_ylabel("Field")
        axs.set_title("Actual Field Shape")
        axs.legend(["bfield", "all peaks", "index_global_max", "baseline"])
        axs.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save the figure
        plt.savefig(f"./field_shapes/field_shape{cycle}.png")
        plt.close(fig)

    return (indexes, peaks), (index_global_max, upline)

            
def find_insertion_point(index_pocket, p_r):
    """
    Finds the insertion point for p_r in a sorted list or numpy array index_pocket.

    Args:
        index_pocket (list or np.ndarray): Sorted list or numpy array of indices.
        p_r (int or float): The value to find the insertion point for.

    Returns:
        int: The insertion index for p_r.
    if isinstance(index_pocket, np.ndarray):
        index_pocket = index_pocket.tolist()  # Convert to list if it's a numpy array
    """

    for i in range(len(index_pocket)):
        if p_r < index_pocket[i]:
            return i  # Insert before index i
    return len(index_pocket)  # Insert at the end if p_r is greater than or equal to all elements


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


""" Integration Methods """

def magnitude(new_vector, prev_vector=[0.0,0.0,0.0]): 
    return np.sqrt(sum([(new_vector[i]-prev_vector[i])*(new_vector[i]-prev_vector[i]) for i in range(len(new_vector))]))

def Ind(i):
  '''
  Indicator Function
              1     if   0 < i < 128
  Ind(i) = {
              1     if   i < 0 or i > 128
  '''         
  # special case is ds = 128
  stat_i = 0<=i<=128

  if stat_i:
    return 1
  else:
    return 0

def four_point_derivative(f, x, h): 
    """
    Approximate the derivative of a function using the 4-point central difference formula.

    Parameters:
    - f: The function to differentiate.
    - x: The point at which to approximate the derivative.
    - h: Step size.
    """
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)

def eul_int(fderiv, dt):
    return fderiv * dt

def rk4_int(const, x, y, z,field_x,field_y,field_z,dt):
    '''
    fderiv: First Derivative  -> Velocity
    x: Position
    '''
    k_1, k_2, k_3, k_4 = 0.0, 0.0, 0.0, 0.0

    k_1 = const * interpolate_vector_field(x, y, z,field_x,field_y,field_z)
    k_2 = const * interpolate_vector_field(x+0.5*k_1[0]*dt, y+0.5*k_1[1]*dt, z + 0.5*k_1[2]*dt,field_x,field_y,field_z)
    k_3 = const * interpolate_vector_field(x+0.5*k_2[0]*dt, y+0.5*k_2[1]*dt, z + 0.5*k_2[2]*dt,field_x,field_y,field_z)
    k_4 = const * interpolate_vector_field(x+k_3[0]*dt, y+k_3[1]*dt, z + k_3[2]*dt,field_x,field_y,field_z)

    return (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

def run_second_order(const, position, velocity, Bx, By, Bz, timestep):
    '''
    fderiv: First Derivative  -> Velocity : np.array
    sderiv: First Derivative  -> Acceleration : np.array
    x,y,z: Position : np.array
    vx,vy,vz: Velocity : np.array
    timestep
    '''
    k_1, k_2, k_3, k_4 = 0.0, 0.0, 0.0, 0.0
    l_1,l_2,l_3,l_4 = 0.0, 0.0, 0.0, 0.0

    l_1, k_1 = const * interpolate_vector_field(position, Bx, By, Bz), velocity
    l_2, k_2 = const * interpolate_vector_field(position[0]+0.5*k_1[0]*timestep, position[1]+0.5*k_1[1]*timestep, position[2] + 0.5*k_1[2]*timestep, Bx, By, Bz), velocity + timestep*l_1/2
    l_3, k_3 = const * interpolate_vector_field(position[0]+0.5*k_2[0]*timestep, position[1]+0.5*k_2[1]*timestep, position[2] + 0.5*k_2[2]*timestep, Bx, By, Bz), velocity + timestep*l_2/2
    l_4, k_4 = const * interpolate_vector_field(position[0]+k_3[0]*timestep, position[1]+k_3[1]*timestep, position[2] + k_3[2]*timestep, Bx, By, Bz), velocity + timestep*l_3

    new_velocity = velocity + timestep * (l_1 + 2 * l_2 + 2 * l_3 + l_4) / 6
    new_position = position + timestep * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    return new_position, new_velocity

"""### Vector Functions"""

def find_enclosing_vectors(i, j, k):
    vectors = []
    neighbors = [
        (0, 0, 0),  # Vertex at the origin
        (0, 0, 1),  # Vertex on the x-axis
        (0, 1, 0),  # Vertex on the y-axis
        (0, 1, 1),  # Vertex on the z-axis
        (1, 0, 0),  # Vertex on the xy-plane
        (1, 0, 1),  # Vertex on the xz-plane
        (1, 1, 0),  # Vertex on the yz-plane
        (1, 1, 1)   # Vertex in all three dimensions
    ]

    point = [np.floor(i), np.floor(j), np.floor(k)]

    for neighbor in neighbors:
        ni, nj, nk = neighbor

        if 0 <= i + ni and 0 <= j + nj and 0 <= k + nk:
            dx = ni + i
            dy = nj + j
            dz = nk + k
            vectors.append(np.array([np.floor(dx), np.floor(dy), np.floor(dz)]))

    return vectors

def interpolate_vector_field(point_i, point_j, point_k, field_x, field_y, field_z):
    u, v, w = point_i - np.floor(point_i), point_j - np.floor(point_j), point_k - np.floor(point_k)

    # Find the eight vectors enclosing the chosen point
    cube = find_enclosing_vectors(point_i, point_j, point_k)

    # Initialize Bp
    interpolated_vector = np.array([0.0, 0.0, 0.0])

    vectors = []

    for c in cube:
        i, j, k = int(c[0]), int(c[1]), int(c[2])

        # Check if indices are within the bounds of the magnetic field arrays
        if 0 <= i < field_x.shape[0] and 0 <= j < field_y.shape[0] and 0 <= k < field_z.shape[0]:

            vectors.append(np.array([field_x[i][j][k], field_y[i][j][k], field_z[i][j][k]]))
            #print([Bx[i][j][k], By[i][j][k], Bz[i][j][k]])
        else:
            # Handle the case where indices are out of bounds (you may choose to do something specific here)
            pass
            #print(f"Indices out of bounds: {i}, {j}, {k}")

    interpolated_vector =  (1 - u) * (1 - v) * (1 - w) * vectors[0] + \
         (1 - u) * w * (1 - v) * vectors[1] + \
         (1 - u) * v * (1 - w) * vectors[2] + \
         (1 - u) * v * w * vectors[3] + \
         u * (1-v) * (1 - w) * vectors[4] + \
         u * (1 - v) * w * vectors[5] + \
         (1 - w) * v * u * vectors[6] + \
         u * v * w * vectors[7]

    return interpolated_vector

def ingrid(i,j,k, index = None):
  '''
  Function finds out if give a ds value for the dimensions of a cube ds x ds x ds grid, a given point i,j,k is inside or outside.
  we want to obtain a boolean function that identifies if a particle is outside of the grid, and the element(s) that are out of bounds
  '''
  # ds = 128 always!
  # True = in the grid, False = out of grid
  stat_i = False if (i < 0 or i > 128) else True
  stat_j = False if (j < 0 or j > 128) else True
  stat_k = False if (k < 0 or k > 128) else True

  vector = [stat_i,stat_j,stat_k]
  if all(vector): # if in grid
    return vector # [True, True, True]
  else: # if not in grid
    index = [not comp for comp in vector] # if vector = [True, False, False], index = [False, True, True] so all out of grid components are True
    return index

"""### Scalar Functions"""

def find_enclosing_scalars(i, j, k):
    scalars = []
    neighbors = [
        (0, 0, 0),  # Vertex at the origin
        (0, 0, 1),  # Vertex on the x-axis
        (0, 1, 0),  # Vertex on the y-axis
        (0, 1, 1),  # Vertex on the z-axis
        (1, 0, 0),  # Vertex on the xy-plane
        (1, 0, 1),  # Vertex on the xz-plane
        (1, 1, 0),  # Vertex on the yz-plane
        (1, 1, 1)   # Vertex in all three dimensions
    ]
    point = [np.floor(i), np.floor(j), np.floor(k)] # three co-ordinates

    for neighbor in neighbors: # let's save all scalars in the 6 co-ordinates
        ni, nj, nk = neighbor # this is a given point

        if 0 <= i + ni and 0 <= j + nj and 0 <= k + nk:
            dx = ni + i
            dy = nj + j
            dz = nk + k
            scalars.append(np.array([np.floor(dx), np.floor(dy), np.floor(dz)]))
    return scalars

def interpolate_scalar_field(point_i, point_j, point_k, scalar_field): 
    u, v, w = point_i - np.floor(point_i), point_j - np.floor(point_j), point_k - np.floor(point_k)

    # Find the eight vectors enclosing the chosen point
    cube = find_enclosing_vectors(point_i, point_j, point_k)

    # Initialize Bp
    interpolated_vector = np.array([0.0, 0.0, 0.0])

    scalars = []

    for c in cube:
        i, j, k = int(c[0]), int(c[1]), int(c[2])

        # Check if indices are within the bounds of the magnetic field arrays
        if 0 <= i < scalar_field.shape[0] and 0 <= j < scalar_field.shape[0] and 0 <= k < scalar_field.shape[0]:

            scalars.append(scalar_field[i][j][k])
            #print(i,j,k,scalar_field[i][j][k])
        else:
            # Handle the case where indices are out of bounds (you may choose to do something specific here)
            pass
            #print(f"Indices out of bounds: {i}, {j}, {k}")
    
    interpolated_scalar = 0.0
    coefficients = [(1 - u) * (1 - v) * (1 - w),  (1 - u) * (1 - v) * w, (1 - u) * v * (1 - w) ,\
                    (1 - u) * v * w , u * (1 - v) * (1 - w),  u * (1 - v) * w, \
                    u * v * (1 - w),  u * v * w] 
    
    for coeff, scalar in zip(coefficients, scalars):
        #print(scalar, coeff)
        interpolated_scalar += coeff * scalar

    return interpolated_scalar



    
def _interpolate_scalar_field(p_i, p_j, p_k, scalar_field, epsilon=1e-10): # implementing Inverse Distance Weighting (Shepard's method)

    # Origin of new RF primed: O' = [0,0,0] but O = [p_i, p_j, p_k] in not primed RF.
    u, v, w = p_i - np.floor(p_i), p_j - np.floor(p_j), p_k - np.floor(p_k)

    # Find the eight coordinates enclosing the chosen point
    coordinates = find_enclosing_scalars(p_i, p_j, p_k)

    # Initialize interpolated_vector
    interpolated_scalar = 0.0
    cum_sum, cum_inv_dis= 0.0, 1.0

    if all(ingrid(p_i,p_j,p_k)):
      for coo in coordinates:
            # coordinates of cube vertices around point    
            i, j, k = int(coo[0])*Ind(int(coo[0])), int(coo[1])*Ind(int(coo[1])), int(coo[2])*Ind(int(coo[2])) # O' position

            # distance from point to vertice of cube
            distance = magnitude([i - p_i, j - p_j, k - p_k]) # real distance
            #print(distance)
          
            # base case
            if distance <= epsilon:
                #print(f"known scalar field at [{p_i},{p_j},{p_k}]")
                return scalar_field[int(p_i)][int(p_j)][int(p_k)]

            cum_sum += scalar_field[i][j][k]/ (distance + epsilon) ** 2
            #print(scalar_field[i][j][k],(distance + 1e-10) ** 2)
            cum_inv_dis += 1 / (distance + epsilon) ** 2 
    else:
       new_coords = [[c[0],c[1],c[2]] for c in coordinates if all(ingrid(c[0],c[1],c[2]))]
       index = ingrid(p_i,p_j,p_k)
       complementary = [[new_coords[im][0]+index[0],new_coords[im][1]++index[1],new_coords[im][2]+index[2]] for im, boo in enumerate(new_coords)]
       coordinates = complementary+new_coords
       for coo in coordinates:
            # coordinates of cube vertices around point    
            i, j, k = int(coo[0]), int(coo[1]), int(coo[2]) # O' position

            # distance from point to vertice of cube
            distance = magnitude([i - p_i, j - p_j, k - p_k]) # real distance          
            #print(scalar_field[i][j][k],(distance + 1e-10) ** 2,np.exp(-distance))
            
            # base case
            if distance <= epsilon:
                #print(f"known scalar field at [{p_i},{p_j},{p_k}]")
                return scalar_field[int(p_i)][int(p_j)][int(p_k)]
            
            cum_sum += scalar_field[i][j][k]*np.exp(-distance**(3)/8)/ (distance + epsilon) ** 2
            cum_inv_dis += 1 / (distance + epsilon) ** 2 

            
    interpolated_scalar = cum_sum / cum_inv_dis
    return interpolated_scalar

"""### Plotting Functions"""

def plot_enclosing_dots(total_time, snaps, field_x, field_y, field_z, p_i, p_j, p_k):

  # Create a meshgrid for the spatial coordinates
  x, y, z = np.meshgrid(np.arange(field_x.shape[0]), np.arange(field_y.shape[1]), np.arange(field_z.shape[2]))

  # Create a 3D plot
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  ax.set_facecolor('black')  # Set background color to black

  # Plotting the magnetic field components
  ax.quiver(x, y, z, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

  # Choose a point in the grid (replace with your desired coordinates)

  timestep = np.linspace(0,total_time, snaps + 1)
  Bp = np.array([0.0,0.0,0.0])
  pos = np.array([0.0,0.0,0.0])

  cube = find_enclosing_vectors(p_i, p_j, p_k)

  for dot in cube:

    ax.scatter(dot[0],dot[1],dot[2], c = 'blue', marker='o')

  ax.quiver(x,y,z, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

  #ax.scatter(xcomp,ycomp,zcomp, c = 'blue', marker='o')
  ax.scatter(p_i, p_j, p_k, c = 'black', marker='x')

  # Set labels and title with white text
  ax.set_xlabel('X', color='white')
  ax.set_ylabel('Y', color='white')
  ax.set_zlabel('Z', color='white')
  ax.set_title('Enclosed Point to Interpolate $B_p$', color='black')

  # Set tick color to white
  ax.tick_params(axis='x', colors='white')
  ax.tick_params(axis='y', colors='white')
  ax.tick_params(axis='z', colors='white')

  # Show the plot
  plt.show()

def plot_3d_vec_field(field_x, field_y, field_z):
    ds = field_x.shape[0]
    x_plot, y_plot, z_plot = np.meshgrid(np.arange(ds), np.arange(ds), np.arange(ds))

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')  # Set background color to black

    # Plotting the magnetic field components
    ax.quiver(x_plot, y_plot, z_plot, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

    # Set labels and title with white text
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title('3D Magnetic Field Visualization', color='black')

    # Set tick color to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Show the plot
    plt.show()

def plot_trajectory(TOTAL_TIME, SNAPSHOTS, POINT_i, POINT_j, POINT_k, field_x, field_y, field_z, k):

    # Create a meshgrid for the spatial coordinates
    x, y, z = np.meshgrid(np.arange(field_x.shape[0]), np.arange(field_y.shape[1]), np.arange(field_z.shape[2]))

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')  # Set background color to black

    # Plotting the magnetic field components
    ax.quiver(x, y, z, field_x, field_y, field_z, length=0.1, normalize=True, color='red', arrow_length_ratio=0.1)

    timestep = np.linspace(0, TOTAL_TIME, SNAPSHOTS + 1)

    trajectory = np.array([[0.0, 0.0, 0.0]])  # Initialize as an empty list

    delta = timestep[1] - timestep[0]
    pos = np.array([POINT_i, POINT_j, POINT_k])

    for time in timestep:

        Bp = interpolate_vector_field(pos[0], pos[1], pos[2], field_x, field_y, field_z)
        pos += k * Bp * delta
        # Update trajectory using np.vstack
        trajectory = np.vstack((trajectory, pos))

    # Plot the trajectory in blue
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', marker='x')

    # Set labels and title with white text
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title('Trajectory Plot in 3D', color='white')

    # Set tick color to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Show the plot
    plt.show()

def multiplot_trajectory_versus_magnitude(domain, legends,  *f ):

    '''
    legends = ["title", "y-coord", "x-coord"]
    '''
    plt.figure(figsize=(15, 5))

    plt.xlabel(legends[2])  # Add an x-label to the axes.
    plt.ylabel(legends[1])   # Add a y-label to the axes.

    plt.scatter(domain, f[0], marker='+', label='B Field Mag')  # Use scatter plot with "+" markers
    plt.scatter(domain, f[1], marker='*', label='CRs Density')  # Use scatter plot with "+" markers

    plt.title(legends[0])       # Add a title to the axes.
    plt.legend([legends[1]])           # Add a legend.
    plt.grid()
    
    min_value = min(f[1])
    max_value = max(f[0]) 

    # Set axis limits based on min and max values
    plt.ylim(min_value, max_value)

    plt.show()

def plot_trajectory_versus_magnitude(line_segments, B_Fields, legends, save_path=None):
    '''
    legends = ["title", "y-coord", "x-coord"]
    '''
    plt.figure(figsize=(15, 5))

    plt.xlabel(legends[2])  # Add an x-label to the axes.
    plt.ylabel(legends[1])   # Add a y-label to the axes.

    plt.scatter(line_segments, B_Fields, marker='+', label='traj')  # Use scatter plot with "+" markers

    plt.title(legends[0])       # Add a title to the axes.
    plt.legend([legends[1]])           # Add a legend.
    plt.grid()
    
    # Calculate min and max values of B_Fields
    if 0.0 in B_Fields:
      B_Fields.remove(0)
    min_value = min(B_Fields)
    max_value = max(B_Fields)

    # Set axis limits based on min and max values
    plt.ylim(min_value, max_value)

    if save_path:
        # Save the plot as PNG or SVG
        plt.savefig(save_path, format='png')  # Use format='svg' for SVG format

    plt.show()

def plot_simulation_data(df):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout(pad = 0)
    plt.show()

