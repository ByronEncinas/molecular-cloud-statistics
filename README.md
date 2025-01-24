# Feature/Update

- [x] python3 arepo_reduction_factor_colors.py [Allocated Mem] [R Boundary] [No. of Lines] [input_snap] [output_name]
- [ ] Cloud Tracker: run for all available files in 'arepo_data/' and keep track of a cube section surrounding the first cloud, and update that value by looking for np.argmax(Density) inside that cube.
- [ ] Gravitational and Thermal Energy Added:

1. **Enclosed Mass \( $M(r)$ \)**:
   $$
   M(r) = \sum_{i=0}^{N} 4 \pi r_i^2 \rho(r_i) \Delta r_i
   $$

2. **Gravitational Potential \( $\phi(r)$ \)**:
   $$
   \phi(r) = \frac{GM(R)}{R} - \sum_{i=j}^N \frac{GM(r_i)}{r_i^2} \Delta r_i
    $$

3. **Gravitational Energy \( $E_{\text{grav}}$ \)**:
   $$
   E_{\text{grav}} = \sum_{i=0}^{N} 4 \pi r_i^2 \rho(r_i) \phi(r_i) \Delta r_i
   $$

4. **Thermal Energy \( U \)**:
   $$
   U = \frac{3}{2} \sum_i P(r_i) \cdot \left( 4 \pi r_i^2 \Delta r \right)
   $$

5. **Magnetic Energy \( U \)**:
$$
E_{\text{magnetic}} = \sum_{i=0}^{N} \frac{B_i^2}{8 \pi} \cdot \left( 4 \pi r_i^2 \Delta r_i \right)
$$