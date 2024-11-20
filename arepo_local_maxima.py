from collections import defaultdict
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import maximum_filter, label
from skimage.measure import regionprops
import glob

FloatType = np.float64
IntType = np.int32


file_list = glob.glob('arepo_data/*.hdf5')
num_file = '430'

for f in file_list:
    if num_file in f:
        filename = f

print(filename)

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
Volume   = Mass/Density
