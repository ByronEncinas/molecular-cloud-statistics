# Feature/Update

- [x] python3 arepo_reduction_factor_colors.py [Allocated Mem] [R Boundary] [No. of Lines] [input_snap] [output_name]
- [x] Cloud Tracker: run for all available files in 'arepo_data/' and keep track of a cube section surrounding the first cloud, and update that value by looking for np.argmax(Density) inside that cube. Now with peak density as marker for time evolution.
- [x] Spherical distribution of x_init variable in region with $n_g > n_{threshold}$
- [ ] Density Threshold Tests: compare reduction factor at different times with different density thresholds 100-50-10 cm$^{-3}$
- Evolution of R(\vec{r})
- [ ] Obtain magnetic field lines in the vicinity of a pocket. Chosse an arbitrary pocket, present in a magnetic field line, match the position of the pocket with a coordinate in space $\vec{r}$ and the direction of $\vec{B}(\vec{r}_{pocket})$ such that we want to generate new points perpendicular to the line.


$$
\{ \vec{r}^i_{new} \}_i^N \perp \vec{B}(\vec{r}_{pocket})
$$
