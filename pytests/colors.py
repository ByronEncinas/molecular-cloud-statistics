import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from scipy import spatial
import sys
import matplotlib
import healpy as hp

FloatType = np.float64
IntType = np.int32


if len(sys.argv)>1:
	zoom=float(sys.argv[1])
else:
	zoom=Boxsize/2

if len(sys.argv)>2:
	start=int(sys.argv[2])
else:
	start=0
	

N=50

dx=4*zoom/N

def get_magnetic_field_at_points(x, Bfield, Bfield_grad, rel_pos):
	n = len(rel_pos[:,0])
	local_fields = np.zeros((n,3))
	for  i in range(n):
		local_fields[i,:] = Bfield[i,:] + np.dot(np.reshape(Bfield_grad[i,:],(3,3)), rel_pos[i,:])
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

def find_points_and_get_fields(x, Bfield, Bfield_grad, Density, Density_grad, Pos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos)
	local_fields = get_magnetic_field_at_points(x, Bfield[cells], Bfield_grad[cells], rel_pos)
	local_densities = get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos)
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	return local_fields, abs_local_fields, local_densities
	
def Heun_step(x, dx, Bfield, Bfield_grad, Density, Density_grad, Pos):
	local_fields_1, abs_local_fields_1, local_densities = find_points_and_get_fields(x, Bfield, Bfield_grad, Density, Density_grad, Pos)
	local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1,(3,1)).T
	x_tilde = x + dx * local_fields_1
	local_fields_2, abs_local_fields_2, local_densities = find_points_and_get_fields(x_tilde, Bfield, Bfield_grad, Density, Density_grad, Pos)
	local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2,(3,1)).T	
	x_final = x + 0.5 * dx * (local_fields_1 + local_fields_2)
	
	return x_final, abs_local_fields_1, local_densities
	
# coordinates => Voronoi position

for i in range(start, start + 1):

	print(i)
	if i<10:
		data = h5py.File('snap_00'+str(i)+'.hdf5', 'r')
	elif 10<=i<100:
		data = h5py.File('snap_0'+str(i)+'.hdf5', 'r')
	else:
		data = h5py.File('snap_'+str(i)+'.hdf5', 'r')

	Boxsize = data['Header'].attrs['BoxSize']	
	Pos = np.array(data['PartType0']['Coordinates'],
		   dtype=FloatType)  # CenterOfMass
	VoronoiPos = np.array(data['PartType0']['Coordinates'], dtype=FloatType)
	Density = np.array(data['PartType0']['Density'], dtype=FloatType)
	#Density_grad = np.array(data['PartType0']['DensityGradient'], dtype=FloatType)
	Density_grad = np.zeros((len(Density),3))
	Bfield = np.array(data['PartType0']['MagneticField'], dtype=FloatType)
	Bfield_grad = np.array(data['PartType0']['BfieldGradient'], dtype=FloatType)
	
	Bfield *= 1/1.496e8 * (1.496e13/1.9885e33)**(-1/2) 
	
	Density *= 1.9885e33 * (1.496e13)**(-3)
	
	Center= 0.5 * Boxsize * np.ones(3)
	
	VoronoiPos+=-Center
	Pos+=-Center
	
	nside = 8        #sets number of cells sampling the spherical boundary layers = 12*nside**2
	rloc  = zoom      #radius of the boundary in code units. 

	npix = 12 * nside**2

	# Add BOLA
	ipix     = np.arange(npix)
	xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix) 
	
	which_up = zz >= 0.0
	
	m = len(zz[which_up])
	
	x_up = np.zeros((m,3))


	x_up[:,0]    = rloc * xx[which_up]
	x_up[:,1]    = rloc * yy[which_up]
	x_up[:,2]    = rloc * zz[which_up]
	
	line=np.zeros((N+1,m,3))
	bfields = np.zeros((N+1,m))
	densities = np.zeros((N+1,m))
	
	line[0,:,:]=x_up
	
	x = x_up
	
	dummy, bfields[0,:], densities[0,:] = find_points_and_get_fields(x, Bfield, Bfield_grad, Density, Density_grad, Pos)
	
	for k in range(N):
		print(k)
		x, bfield, dens = Heun_step(x, dx, Bfield, Bfield_grad, Density, Density_grad, VoronoiPos)
		
		line[k+1,:,:] = x
		bfields[k+1,:] = bfield
		densities[k+1,:] = dens

	which_down = zz < 0.0
	
	n = len(zz[which_down])
	
	x_down = np.zeros((n,3))


	x_down[:,0]    = rloc * xx[which_down]
	x_down[:,1]    = rloc * yy[which_down]
	x_down[:,2]    = rloc * zz[which_down]
	
	line_rev=np.zeros((N+1,n,3))
	bfields_rev = np.zeros((N+1,n))
	densities_rev = np.zeros((N+1,n))
	
	line_rev[0,:,:]=x_down
	
	x = x_down
	
	dummy, bfields_rev[0,:], densities_rev[0,:] = find_points_and_get_fields(x, Bfield, Bfield_grad, Density, Density_grad, Pos)
	
	for k in range(N):
		print(k)
		x, bfield, dens = Heun_step(x, - dx, Bfield, Bfield_grad, Density, Density_grad, VoronoiPos)
		
		line_rev[k+1,:,:] = x
		bfields_rev[k+1,:] = bfield
		densities_rev[k+1,:] = dens
			
	ax = plt.figure().add_subplot(projection='3d')
	
	dens_min = np.log10(min(np.min(densities), np.min(densities_rev)))
	dens_max = np.log10(max(np.max(densities), np.max(densities_rev)))
	
	dens_diff = dens_max - dens_min
	
	for k in range(m):
		x=line[:,k,0]
		y=line[:,k,1]
		z=line[:,k,2]
		
		which = x**2 + y**2 + z**2 <= zoom**2
		
		x=x[which]
		y=y[which]
		z=z[which]
		
		for l in range(N):
			#ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color = 'blue' , alpha = (np.log10(0.5 * (densities[l,k] + densities[l+1,k])) - dens_min) / dens_diff,linewidth=0.1)
			ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color = ((np.log10(0.5 * (densities[l,k] + densities[l+1,k])) - dens_min) / dens_diff , 0.0 , 1.0 - (np.log10(0.5 * (densities[l,k] + densities[l+1,k])) - dens_min) / dens_diff),linewidth=0.1)

	for k in range(n):
		x=line_rev[:,k,0]
		y=line_rev[:,k,1]
		z=line_rev[:,k,2]
		
		which = x**2 + y**2 + z**2 <= zoom**2
		
		x=x[which]
		y=y[which]
		z=z[which]

		for l in range(N):
			ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color = ((np.log10(0.5 * (densities_rev[l,k] + densities_rev[l+1,k])) - dens_min) / dens_diff , 0.0 , 1.0 - (np.log10(0.5 * (densities_rev[l,k] + densities_rev[l+1,k])) - dens_min) / dens_diff),linewidth=0.1)
				
	ax.set_xlim(-zoom,zoom)
	ax.set_ylim(-zoom,zoom)
	ax.set_zlim(-zoom,zoom)
	ax.set_xlabel('x [AU]')
	ax.set_ylabel('y [AU]')
	ax.set_zlabel('z [AU]')
	ax.set_title('Magnetic field morphology')
	
	plt.show()
	#plt.savefig('Bfield_ohm_alt.png',bbox_inches='tight')
