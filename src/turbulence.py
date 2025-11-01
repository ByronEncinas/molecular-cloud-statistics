# Contains functions to calculate the turbulence, eddy size, velocity fluctuation
# velocity dispersion, etc.

# also start using type hinting
import numpy as np

# all in cgs
electron_charge = 1.0
light_speed = 1.0
electron_mass = 1.0
permeability_free_space = 1.0

def velocity_stats(Velocity: np.array):
    return np.mean(Velocity), Velocity - np.mean(Velocity), np.std(Velocity)

def magnetic_diffusivity(Density: np.array, CollisionFrec: np.float) -> None:
    sigma0 = Density*electron_charge*electron_charge/(electron_mass*CollisionFrec)
    return (permeability_free_space*sigma0)**(-1)



