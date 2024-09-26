#!/bin/bash

# this will create arepo_pockets plots and histograms
python3 arepo_reduction_factor_colors.py 300 0.1 100 > red_factor.out &

# this will create arepo_npys, arepo_pockets and 2x2 mosaic plots
python3 arepo_get_field_lines_colors.py  300 0.1 100 > get_lines.out &

# this will create directories as many as files are available.
python3 arepo_density_profile.py 300 > profile.out &
