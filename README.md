# Simulation Configuration for Cosmic Ray and Plasma Research

This repository provides scripts for running simulations related to cosmic rays and plasma behavior. The scripts allow users to easily configure simulation parameters via command-line arguments. Below is an overview of how to use the scripts and customize their settings.

# images/

Contains graphs related to the steps and calculation of the ionization rate using simulated magnetic field lines

# Padovani.py

Uses heuristic methods to calculate the ionization rate taking into the account magnetic focusing and mirroring using simulated magnetic field lines

# stats.py

Calculates both column densities along CR path (field lines) and line of sight according to 'threshold' density and 'dense_core' as parameters to halt line tracing. Uses Padovani et al (2015) fit for the ionization rate to compare the impact of these two paradigms on column density measurements and models.

# clouds.py

Finds and stores position of isolated density peaks in a simulation snapshot.

# library.py

Contains constants, interpolation functions and analysis tools.

# other files

To be specified