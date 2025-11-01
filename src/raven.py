import os, sys, time, glob, logging
import numpy as np, h5py
from scipy import stats
import yt; from yt import units as u
import os
from library import *

def main(): 
    # reference start time
    start_time = time.time()

    # Set yt logging to show only warnings and errors
    yt.funcs.mylog.setLevel(logging.WARNING)

    FloatType = np.float64
    IntType = np.int32

    gr_cm3_to_nuclei_cm3 = 6.02214076e+23 / (2.35) * 6.771194847794873e-23  # Wilms, 2000 ; Padovani, 2018 ism mean molecular weight is # conversion from g/cm^3 to nuclei/cm^3

    input_case = 'ideal'
    how_many   = 10

    if input_case == 'ideal':
        file_list = sorted(glob.glob('./arepo_data/ideal_mhd/*.hdf5'))
    elif input_case == 'amb':
        file_list = sorted(glob.glob('./arepo_data/ambipolar_diffusion/*.hdf5'))

    centers = np.zeros((len(file_list),how_many,3))
    centers_curr = np.zeros((how_many,3))

    print(file_list)
    with open(f"./cloud.txt", "w") as file:
        file.write("snap,cloud,time_value,c_coord_X,c_coord_Y,c_coord_Z,Peak_Density\n")

    for fileno, filename in enumerate(file_list[::-1]):

        data = h5py.File(filename, 'r')
        header_group = data['Header']
        time_value = header_group.attrs['Time']
        snap = filename.split('/')[-1].split('.')[0][-3:]
        Boxsize = data['Header'].attrs['BoxSize']
        Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
        Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)

        #dendogram_analysis(Density*gr_cm3_to_nuclei_cm3)

        xc = Pos[:, 0]
        yc = Pos[:, 1]
        zc = Pos[:, 2]

        region_radius = 2

        if fileno == 0:
            for _each in range(how_many):
                # here we find the clouds
                c_arg = np.argmax(Density)
                c_coord = Pos[c_arg, :]
                peak_density = Density[c_arg]*gr_cm3_to_nuclei_cm3 # in cm-3
                c = ((xc - c_coord[0])**2 + (yc - c_coord[1])**2 + (zc - c_coord[2])**2 < region_radius**2)
                Density[c] *= 0.0 # remove the cloud to search for others
                centers[fileno,_each,:] = c_coord.copy() # save
            #print(*centers[fileno,0,:], *centers[fileno,1,:])

                with open(f"./cloud.txt", "a") as file:
                    print(f"{snap},cloud-{_each},{time_value},{c_coord[0]},{c_coord[1]},{c_coord[2]},{peak_density},{np.log10(peak_density)}")
                    file.write(f"{snap},cloud-{_each},{time_value},{c_coord[0]},{c_coord[1]},{c_coord[2]},{peak_density}\n")
            continue

        # Nice! we have 'how_many' coordinates for clouds in the 499th snapshot
        # follow the clouds we already have
        #for _each, _center in enumerate(centers):

        for _each in range(how_many):
            # isolate previous coordinates of cloud
            _center = centers[fileno-1, _each,:]

            # isolate the nearby cells to catch the evolutions
            c = ((xc - _center[0])**2 + (yc - _center[1])**2 + (zc - _center[2])**2 < region_radius**2)
            # here we isolate the region around the cloud in order to find the new position
            subset_indices = np.where(c)[0]
            c_arg_local = np.argmax(Density[c])
            c_arg_global = subset_indices[c_arg_local]

            peak_density = Density[c_arg_global] * gr_cm3_to_nuclei_cm3
            c_coord = Pos[c_arg_global, :]
            Density[c] *= 0.0
            centers_curr[_each,:]= c_coord.copy()
            with open(f"./cloud.txt", "a") as file:
                print(f"{snap},cloud-{_each},{time_value},{c_coord[0]},{c_coord[1]},{c_coord[2]},{peak_density},{np.log10(peak_density)}")
                file.write(f"{snap},cloud-{_each},{time_value},{c_coord[0]},{c_coord[1]},{c_coord[2]},{peak_density}\n")
        centers[fileno,:,:] = centers_curr.copy()

if __name__=='__main__':

    import pandas as pd
    from pandas.plotting import table

    dfi = pd.read_csv(f'./ideal_clouds.txt')     
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    for x in range(6):
        df_i = dfi.iloc[x::6][::-1]
        t_i = df_i['time_value'].to_numpy()
        dt_i = df_i['time_value'].diff()
        vel_i = np.sqrt(df_i['c_coord_X'].diff()**2 + 
                        df_i['c_coord_Y'].diff()**2 + 
                        df_i['c_coord_Z'].diff()**2) / dt_i .to_numpy()
        
        row = x // 3
        col = x % 3

        axs[row, col].plot(t_i, vel_i, '.-', color='blue', alpha=0.4, label='Velocity')
        axs[row, col].plot(t_i,t_i*vel_i.mean()/t_i , '--', color='black', label='Mean')
        axs[row, col].text(0.05, 0.85, f'Mean Velocity: {pc_myrs_to_km_s*vel_i.mean():.5f} km/s', transform=axs[row, col].transAxes, 
                        fontsize=18, verticalalignment='top', horizontalalignment='left', color='black')

        axs[row, col].set_xlabel('$t - t_{G-ON}$ [Myrs]', fontsize=14)
        axs[row, col].set_ylabel('Velocity [pc/Myrs]', fontsize=14)
        axs[row, col].set_title(f'Cloud {x+1}', fontsize=16)
        axs[row, col].tick_params(axis='both', which='major', labelsize=10)
        axs[row, col].grid(True)
        axs[row, col].legend(loc='upper right', fontsize =16)

    plt.tight_layout()
    plt.savefig('images/raven/vel_mosaic.png', dpi=300, bbox_inches='tight')
    plt.close()

    """.2. Shock wave (from: https://www.aanda.org/articles/aa/pdf/2015/08/aa25907-15.pdf)
            In order to investigate the effects of diffusive shock acceleration, Padian was extended to model a one-dimensional MHD
        shock front: The ambient plasma will flow through the shock rest frame with a defined velocity toward the shock front. At 
        the position of the shock front, a discontinuity in the magnetic field and ambient gas velocity is added according to the Rankine-
        Hugoniot jump conditions (see, e.g., Schlickeiser 2002, Chap. 16 and Longair 2011, Chap. 11). Note that the back reaction of the
        cosmic-ray particles on the shock wave will be neglected (cf. Riquelme & Spitkovsky 2010)
    """    
    # sudden jumps in the velocity may be studied by analizing shock waves, turbulence, etc. 
    # But I gotta learn how to identify them first.


    dfn = pd.read_csv(f'./amb_clouds.txt')     
    df_n0 = dfi.iloc[::6]
    
    # plot diff(t_ideal) and diff(t_non)
    # plot n_ig(t) and n_ng(t)
    # average velocity of each cloud

    tn = df_n0['time_value'].iloc[::-1]
    ni = df_n0['Peak_Density'].iloc[::-1]
    dtn = tn.diff() 

    fig, ax = plt.subplots() # Add figsize for better display

    ax.plot(tn, dtn,'.-', color='blue',label='Time Step ($\Delta t$)')    
    ax.set_xlabel('$t - t_{G-ON}$ [Myrs]', fontsize=16)
    ax.set_ylabel('Time Step [Myrs]', fontsize=16) 
    #ax.legend(fontsize=14) 
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='y', colors='blue', labelsize=12)
    ax.set_xlim(0.4,2.05)

    ax_ = ax.twinx()
    ax_.plot(tn, ni,'.-', color='red', label='Time Step ($\Delta t$)')    

    ax_.set_xlabel('$t - t_{G-ON}$ [Myrs]', fontsize=16)

    ax_.set_yscale('log') 
    ax_.tick_params(axis='y', which='major', labelsize=12)
    #ax_.legend(fontsize=14) 
    ax_.spines['right'].set_color('red')
    ax_.set_ylabel('Density [cm$^{-3}$]', color='red', fontsize=16)
    ax_.tick_params(axis='y', colors='red', labelsize=12)

    plt.title('Distribution of Time Steps', fontsize=16)
    plt.tight_layout()
    plt.grid(True) 

    plt.savefig('images/raven/t.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # --- Subplot 0 ---
    for _c in range(6):
        dfi.iloc[_c::6].plot(x='time_value', y='Peak_Density', logy=True, ax=ax0, label=f'cloud-{_c}')
    ax0.set_xlabel('$t - t_{G-ON}$ [Myr]', fontsize=16)
    ax0.set_ylabel('Density [cm$^{-3}$]', fontsize=16)
    ax0.set_title('Ideal MHD', fontsize=16)
    ax0.legend(fontsize=16)
    ax0.grid(True)

    # --- Subplot 1 ---
    for _c in range(6):
        dfn.iloc[_c::6].plot(x='time_value', y='Peak_Density', logy=True, ax=ax1, label=f'cloud-{_c}')

    ax1.set_xlabel('$t - t_{G-ON}$ [Myr]', fontsize=16)
    ax1.set_title('Non-ideal MHD - AD', fontsize=16)
    ax1.legend(fontsize=16)
    ax1.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for legend
    plt.savefig('images/raven/c_mos.png', dpi=300, bbox_inches='tight')
    plt.close()
