import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy import spatial
import healpy as hp
import numpy as np
import time
import matplotlib
import random
import copy
import h5py
import os
import sys

from library import *

FloatType = np.float64
IntType = np.int32

"""
Start of the code 

python3 colors.py [zoom_boundary] [time-snap]

- [zoom_boundary] default is undefined
	will perform test with several values to 
- [time-snap] default is 0.0

Volume of a Voronoi Cell

SurfaceArea 	AREA 	UnitLength2 a**2 h**−2 Surface area of a Voronoi cell (OUTPUT_SURFACE_AREA)
NumFacesCell 	NFAC 	Adimensonal            Number of faces of a Voronoi cell (OUTPUT_SURFACE_AREA)

Volume = ?

python3 arepo_field_lines.py [Number of points] [rloc_boundary] [rloc_center]

"""
	

"""
Parameters

- [N] default is 50 as the total number of steps in the simulation
- [dx] default 4/N of the zoom_boundary (radius of spherical region of interest) variable

"""
FloatType = np.float64
IntType = np.int32

if len(sys.argv)>2:
	# first argument is a number related to zoom_boundary
	N=int(sys.argv[1])
	zoom_boundary=float(sys.argv[2])
	zoom_center=float(sys.argv[3])
else:
	N            =100
	zoom_boundary=80   # rloc_boundary for boundary region of the cloud
	zoom_center  =1      # rloc_boundary for inner region of the cloud

pp = sys.argv[-1]

filename = 'arepo_data/snap_430.hdf5'


start_time=time.time()

"""
Functions/Methods

- Data files provided do not contain  
- 
"""
def magnitude(new_vector, prev_vector=[0.0,0.0,0.0]): 
    return np.sqrt(sum([(new_vector[i]-prev_vector[i])*(new_vector[i]-prev_vector[i]) for i in range(len(new_vector))]))

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

def find_points_and_relative_positions(x, Pos):
	dist, cells = spatial.KDTree(Pos[:]).query(x, k=1,workers=12)
	rel_pos = VoronoiPos[cells] - x
	return dist, cells, rel_pos

def find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos)
	local_fields = get_magnetic_field_at_points(x, Bfield[cells], rel_pos)
	local_densities = get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos)
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	
	return local_fields, abs_local_fields, local_densities, cells
	
def Heun_step(x, dx, Bfield, Density, Density_grad, Pos):

	local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos)
	local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T
		
	if dx > 0:
		dx = 0.4*((4/3)*Volume[cells][0]/np.pi)**(1/3)
	else:
		dx = -0.4*((4/3)*Volume[cells][0]/np.pi)**(1/3)

	x_tilde = x + dx * local_fields_1
	local_fields_2, abs_local_fields_2, local_densities, cells = find_points_and_get_fields(x_tilde, Bfield, Density, Density_grad, Pos)
	local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
	abs_sum_local_fields = np.sqrt(np.sum((local_fields_1 + local_fields_2)**2,axis=1))

	unitary_local_field = (local_fields_1 + local_fields_2) / np.tile(abs_sum_local_fields, (3,1)).T
	x_final = x + 0.5 * dx * unitary_local_field[0]
	
	x_final = x + 0.5 * dx * (local_fields_1 + local_fields_2)
	
	return x_final, abs_local_fields_1, local_densities

"""  B. Jesus Velazquez One Dimensional """

data = h5py.File(filename, 'r')

print(filename, "Loaded")

Boxsize = data['Header'].attrs['BoxSize'] #

VoronoiPos = np.array(data['PartType0']['Coordinates'], dtype=FloatType) # Voronoi Point in Cell
Pos = np.array(data['PartType0']['CenterOfMass'], dtype=FloatType)  # CenterOfMass in Cell
Bfield = np.array(data['PartType0']['MagneticField'], dtype=FloatType)
Bfield_grad  = np.zeros((len(Bfield),3))
Density = np.array(data['PartType0']['Density'], dtype=FloatType)
Density_grad = np.zeros((len(Density),3))
Mass = np.array(data['PartType0']['Masses'], dtype=FloatType)
Volume = Mass/Density


# printing relevant info about the data

Bfield  *= 1# 1/1.496e8 * (1.496e13/1.9885e33)**(-1/2) # in cgs
Density *= 1#1.9885e33 * (1.496e13)**(-3)				# in cgs
Mass    *= 1#1.9885e33
Volume   = Mass/Density

#Center= 0.5 * Boxsize * np.ones(3) # Center
#Center = np.array( [91,       -110,          -64.5]) #117
#Center = np.array( [96.6062303,140.98704002, 195.78020632]) #117
Center = Pos[np.argmax(Density),:] #430

# Make Box Centered at the Point of Interest or High Density Region

# all positions are relative to the 'Center'
VoronoiPos-=Center
Pos-=Center

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

# idk how but this picks the regions to be worked on	
nside = 1      # sets number of cells sampling the spherical boundary layers = 12*nside**2
npix  = 1 #12 * nside ** 2 

rloc_boundary  = zoom_boundary      # radius of the boundary in cgs units. (zoom_boundary is of the order or less of the Boxsize)
rloc_center    = zoom_center		# radius of sphere at the center of cloud (order of 1% of zoom_boundary)

# Add BOLA (slicing the spherical region we want to work with)

ipix_center       = np.arange(npix)
#print(ipix_center)

xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix_center)
#print(xx)

print("Steps in Simulation: ", N)
print("rloc_boundary      : ", zoom_boundary)
print("rloc_center        : ", zoom_center)
print("\nBoxsize: ", Boxsize) # 256
print("Center: ", Center) # 256
print("Position of Max Density: ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume: ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume: ", Volume[np.argmax(Volume)],"\n") # 256

if True:
	nside = 8     # sets number of cells sampling the spherical boundary layers = 12*nside**2
	npix  = 12 * nside ** 2 
	ipix_center       = np.arange(npix)
	xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix_center)
	xx = np.array(random.sample(sorted(xx),1))
	yy = np.array(random.sample(sorted(yy),1))
	zz = np.array(random.sample(sorted(zz),1))

m = len(zz) # amount of values that hold which_up_down

x_init = np.zeros((m,3))
x_init[:,0]      = rloc_center * xx[:]
x_init[:,1]      = rloc_center * yy[:]
x_init[:,2]      = rloc_center * zz[:]

line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
bfields   = np.zeros((N+1,m))
densities = np.zeros((N+1,m))

line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
bfields_rev = np.zeros((N+1,m))
densities_rev = np.zeros((N+1,m))

line[0,:,:]     =x_init
line_rev[0,:,:] =x_init

x = x_init

print(x_init[0,:])

dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos)
dummy, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos)

# propagates from same inner region to the outside in -dx direction
for k in range(N):
	
	dx = 1.0
	x, bfield, dens = Heun_step(x, dx, Bfield, Density, Density_grad, VoronoiPos)
	
	line[k+1,:,:] = x
	bfields[k+1,:] = bfield
	densities[k+1,:] = dens

# propagates from same inner region to the outside in -dx direction

for k in range(N):
	x, bfield, dens = Heun_step(x, -dx, Bfield, Density, Density_grad, VoronoiPos)
	
	line_rev[k+1,:,:] = x
	bfields_rev[k+1,:] = bfield
	densities_rev[k+1,:] = dens

line_rev = line_rev[1:,:,:]
bfields_rev = bfields_rev[1:,:] 
densities_rev = densities_rev[1:,:]

dens_min = np.log10(min(np.min(densities),np.min(densities_rev)))
dens_max = np.log10(max(np.max(densities),np.max(densities_rev)))

dens_diff = dens_max - dens_min

path           = np.append(line, line_rev, axis=0)
path_bfields   = np.append(bfields, bfields_rev, axis=0)
path_densities = np.append(densities, densities_rev, axis=0)

for j, _ in enumerate(path[0,:,0]):
	# for trajectory 
	radius_vector      = np.zeros_like(path[:,j,:])
	magnetic_fields    = np.zeros_like(path_bfields[:,j])
	gas_densities      = np.zeros_like(path_densities[:,j])
	trajectory         = np.zeros_like(path[:,j,0])

	prev_radius_vector = path[0,j,:]
	diff_rj_ri = 0.0

	for k, pk in enumerate(path[:,j,0]):
		
		radius_vector[k]    = path[k,j,:]
		magnetic_fields[k]  = path_bfields[k,j]
		gas_densities[k]    = path_densities[k,j]
		diff_rj_ri = magnitude(radius_vector[k], prev_radius_vector)
		trajectory[k] = trajectory[k-1] + diff_rj_ri
		#print(radius_vector[k], magnetic_fields[k], gas_densities[k], diff_rj_ri)
		
		prev_radius_vector  = radius_vector[k] 

	trajectory[0] *= 0.0

time_end = time.time()-start_time
elapsed_time_minutes= time_end/60

np.save("arepo_output_data/ArePositions.npy", radius_vector)
np.save("arepo_output_data/ArepoTrajectory.npy", trajectory)
np.save("arepo_output_data/ArepoNumberDensities.npy", gas_densities)
np.save("arepo_output_data/ArepoMagneticFields.npy", magnetic_fields)


if True:
	# Create a figure and axes for the subplot layout
	fig, axs = plt.subplots(2, 1, figsize=(8, 6))

	axs[0].plot(trajectory, magnetic_fields, linestyle="--", color="m")
	axs[0].scatter(trajectory, magnetic_fields, marker="+", color="m")
	axs[0].set_xlabel("trajectory (cgs units Au)")
	axs[0].set_ylabel("$B(s)$ (cgs units )")
	axs[0].set_title("Individual Magnetic Field Shape")
	axs[0].legend()
	axs[0].grid(True)

	axs[1].plot(trajectory, gas_densities, linestyle="--", color="m")
	axs[1].set_xlabel("trajectory (cgs units Au)")
	axs[1].set_ylabel("$n_g(s)$ Field (cgs units $M_{sun}/Au^3$) ")
	axs[1].set_title("Gas Density along Magnetic Lines")
	axs[1].legend()
	axs[1].grid(True)

	# Adjust layout to prevent overlap
	plt.tight_layout()

	# Save the figure
	plt.savefig("field_shapes/field-density_shape.png")

	# Show the plot
	#plt.show()
	plt.close(fig)

	ax = plt.figure().add_subplot(projection='3d')

	for k in range(m):
		print(k)
		x=path[:,k,0] # 1D array
		y=path[:,k,1]
		z=path[:,k,2]
		
		which = x**2 + y**2 + z**2 <= zoom_boundary**2
		
		x=x[which]
		y=y[which]
		z=z[which]
		
		for l in range(len(z)):
			ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color="m",linewidth=0.3)

		#ax.scatter(x_init[0], x_init[1], x_init[2], marker="v",color="m",s=10)
		ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
		ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
		

	ax.set_xlim(-zoom_boundary,zoom_boundary)
	ax.set_ylim(-zoom_boundary,zoom_boundary)
	ax.set_zlim(-zoom_boundary,zoom_boundary)
	ax.set_xlabel('x [AU]')
	ax.set_ylabel('y [AU]')
	ax.set_zlabel('z [AU]')
	ax.set_title('From Core to Outside in +s, -s directions')

	#plt.savefig(f'field_shapes/MagneticFieldThreading.png',bbox_inches='tight')

	#plt.close()
	#plt.show()

print("Elapsed Time (Minutes): ", elapsed_time_minutes)