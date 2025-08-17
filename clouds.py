import os, sys, time, glob, logging
import numpy as np, h5py, seaborn as sns
from scipy import stats
import yt; from yt import units as u
from library import *

# Set yt logging to show only warnings and errors
yt.funcs.mylog.setLevel(logging.WARNING)

FloatType = np.float64
IntType = np.int32

try:
	input_snap = str(sys.argv[1])
	how_many   = int(sys.argv[2])
except:
    input_snap = '430'
    how_many   = 5

file_list = [sorted(glob.glob('arepo_data/ideal_mhd/*.hdf5')), sorted(glob.glob('arepo_data/ambipolar_diffusion/*.hdf5'))]

for list in file_list:
    for fileno, filename in enumerate(list[::-1]):
        data = h5py.File(filename, 'r')
        header_group = data['Header']
        time_value = header_group.attrs['Time']
        snap = filename.split('/')[-1].split('.')[0][-3:]
        if input_snap != snap:
            break

        if 'amb' in filename:
            typpe = 'amb'
        else:
            typpe = 'ideal'
        parent_folder = "clouds/"+typpe 
        children_folder = os.path.join(parent_folder, snap)
        os.makedirs(children_folder, exist_ok=True)
        Boxsize = data['Header'].attrs['BoxSize']
        #VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
        Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
        Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
        #Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
        
        xc = Pos[:, 0]
        yc = Pos[:, 1]
        zc = Pos[:, 2]

        region_radius = 10

        ds = yt.load(filename)

    
        for each in range(how_many):

            if each == 0:
                CloudCord = Pos[np.argmax(Density), :]
                cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)
                Density[cloud_sphere] *= 0.0
                PeakDensity = np.log10(Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3)
                with open(f"clouds/{typpe}_clouds.txt", "w") as file:
                    file.write("snap,index,CloudCord_X,CloudCord_Y,CloudCord_Z,Peak_Density\n")
                    file.write(f"{snap},{each},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")
                Density[cloud_sphere] = 0.0

                sp = yt.SlicePlot(
                    ds, 
                    'z', 
                    ('gas', 'density'), 
                    center=[CloudCord[0], CloudCord[1], CloudCord[2]],
                    width=256* u.pc
                )
                sp.annotate_sphere(
                    [CloudCord[0], CloudCord[1], CloudCord[2]],  # Center of the sphere
                    radius=(region_radius*0.5, "pc"),  # Radius of the sphere (in physical units, e.g., pc)
                    circle_args={"color": "black", "linewidth": 2}  # Styling for the sphere
                )
            
            else:
                cloud_sphere = ((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2 < region_radius**2)
                Radius = np.sqrt((xc - CloudCord[0])**2 + (yc - CloudCord[1])**2 + (zc - CloudCord[2])**2)
                Density[cloud_sphere] = 0.0

                CloudCord = Pos[np.argmax(Density), :]
                PeakDensity = np.log10(Density[np.argmax(Density)]*gr_cm3_to_nuclei_cm3)

                with open(f"clouds/{typpe}_clouds.txt", "a") as file:
                    file.write(f"{snap},{each},{CloudCord[0]},{CloudCord[1]},{CloudCord[2]},{PeakDensity}\n")



                sp.annotate_sphere(
                    [CloudCord[0], CloudCord[1], CloudCord[2]],  # Center of the sphere
                    radius=(region_radius*0.5, "pc"),  # Radius of the sphere (in physical units, e.g., pc)
                    circle_args={"color": "black", "linewidth": 2}  # Styling for the sphere
                )

            sp.annotate_text(
                [CloudCord[0], CloudCord[1], CloudCord[2]],  # Slightly offset to avoid overlap
                text=f"{each}",  # Your custom label
                coord_system="data",  # Use data coordinates
                text_args={"color": "black", "fontsize": 12}
            )

            sp.annotate_timestamp(redshift=False)

            sp.save(os.path.join(parent_folder, f"{typpe}_{snap}_{each}.png"))