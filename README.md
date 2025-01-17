# Feature/Update

- [x] python3 arepo_reduction_factor_colors.py [Allocated Mem] [R Boundary] [No. of Lines] [input_snap] [output_name]
- [ ] Cloud Tracker: run for all available files in 'arepo_data/' and keep track of a cube section surrounding the first cloud, and update that value by looking for np.argmax(Density) inside that cube.
- [ ] Gravitational Energy Added:


# Gravitational Energy Calculation

This document describes the numerical implementation of gravitational energy for a gas cloud using a given density distribution as a function of radius \( r \). The gravitational energy is calculated using the formula:

```markdown
E_{	ext{grav}} = \int_0^R 4 \pi r^2 ar{
ho}(r) \phi(r) \, dr
```

where:
- \( ar{
ho}(r) \) is the average density at radius \( r \),
- \( \phi(r) \) is the gravitational potential at radius \( r \), given by:

```markdown
\phi(r) = rac{GM(R)}{R} - \int_r^R rac{GM(r)}{r^2} \, dr
```

where \( M(r) \) is the cumulative mass enclosed within radius \( r \), defined as:

```markdown
M(r) = \int_0^r 4 \pi r^2 ar{
ho}(r) \, dr
```

## Numerical Implementation

**Radial Positions \( r \)**: 

The radial positions of gas shells are computed as the norm of the position vectors:

```markdown
r = \| x \|
```

where \( x \) contains the coordinates of each shell.

In Python:

```python
rad = np.linalg.norm(x[:, :], axis=1)
```

**Enclosed Mass \( M(r) \)**:

The cumulative enclosed mass is calculated incrementally:

```markdown
M(r) = \sum_{i=0}^{N} 4 \pi r_i^2 
ho(r_i) \Delta r_i
```

where \( \Delta r_i \) is the thickness of the \( i \)-th shell.

In Python:

```python
M_r = np.cumsum(4 * np.pi * dens * rad**2 * dx_vec * parsec_to_cm3)
```

**Gravitational Potential \( \phi(r) \)**:

The gravitational potential at radius \( r \) is computed as:

```markdown
\phi(r) = rac{GM(R)}{R} - \sum_{i=j}^N rac{GM(r_i)}{r_i^2} \Delta r_i
```

where \( R \) is the outermost radius and \( j \) is the index of the current shell.

In Python:

```python
phi = (G * M_r[-1] / rad[-1]) - np.cumsum(G * M_r / rad**2)
```

**Gravitational Energy \( E_{	ext{grav}} \)**:

The gravitational energy is calculated as:

```markdown
E_{	ext{grav}} = \sum_{i=0}^{N} 4 \pi r_i^2 
ho(r_i) \phi(r_i) \Delta r_i
```

In Python:

```python
grav_energy_density = 4 * np.pi * rad**2 * dens * phi
energy_grav[k + 1, :] = np.cumsum(grav_energy_density * dx_vec)
```

## Additional Energies

**Magnetic Energy**:

```markdown
E_{	ext{magnetic}} = rac{B^2}{8 \pi} \cdot 	ext{Volume}
```

In Python:

```python
energy_magnetic[k + 1, :] = bfield * bfield / (8 * np.pi) * vol
```

**Thermal Energy**:

```markdown
E_{	ext{thermal}} = rac{3}{2} k_B T
```

where \( T \) is the temperature of the gas.

In Python:

```python
energy_thermal[k + 1, :] = (3 / 2) * boltzmann_constant_cgs * temp
```

## Numerical Integration Notes

**Shell Thickness \( \Delta r_i \)**:

```markdown
\Delta r_i = \left( rac{3}{4 \pi 	ext{Volume}} 
ight)^{1/3}
```

ensures uniform shell integration.

**Cumulative Integration**: The use of `np.cumsum` provides efficient and stable numerical summation over discrete shells.

## Code Variables

| Variable          | Description                                |
|-------------------|--------------------------------------------|
| \( x \)           | Position vectors of the gas shells.        |
| \( 	ext{rad} \)  | Radial distances of the gas shells.        |
| \( 	ext{dens} \) | Density distribution as a function of radius. |
| \( 	ext{vol} \)  | Volume of each shell.                      |
| \( 	ext{dx\_vec} \) | Thickness of each shell.                   |
| \( M_r \)         | Enclosed mass as a function of radius.     |
| \( \phi \)        | Gravitational potential at each radius.    |
| \( 	ext{energy\_grav} \) | Gravitational energy for each shell.   |
| \( 	ext{energy\_magnetic} \) | Magnetic energy for each shell. |
| \( 	ext{energy\_thermal} \) | Thermal energy for each shell.   |

This implementation ensures efficient integration and provides physical insights into the system's energy dynamics.
