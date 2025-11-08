# Simulation Configuration for Cosmic Ray in Plasma Environments

Structure of the code

magnetic_pockets/
└── molecular-cloud-statistics/
    ├── README.md
    ├── requirements.txt
    ├── data.py
    ├── stats.py
    ├── timeseries.py
    │
    ├── util/
    │   ├── ideal_clouds.txt
    │   ├── amb_clouds.txt
    │   ├── ideal_cloud_trajectory.txt
    │   └── amb_cloud_trajectory.txt
    │
    ├── src/
    │   ├── clouds.py
    │   ├── info.py
    │   ├── library.py
    │   ├── padovani.py
    │   ├── parse.py
    │   ├── pool.map.py
    │   ├── raven.py
    │   ├── tracker.py
    │   └── turbulence.py
    │
    ├── series/
    │   ├── *.pkl
    │   └── *.png
    │
    ├── images/
    │   ├── columns/*.png
    │   ├── descriptor/*.png
    │   ├── i_rate/*.png
    │   ├── raven/*.png
    │   ├── reduction/*.png
    │   └── xyz_distro/*.png
    │
    ├── gists/
    │   ├── 3dhist.py
    │   ├── advanced.md
    │   ├── alexei.py
    │   ├── analysis.py
    │   ├── arepo_get_field_lines.py
    │   ├── arepo_get_field_lines_parallel.py
    │   ├── colors.py
    │   ├── crutcher.py
    │   ├── D3color.py
    │   ├── divergence.py
    │   ├── fix.py
    │   ├── gabriel.py
    │   ├── io.py
    │   ├── ked.py
    │   ├── loop_stats.py
    │   ├── los_stats.py
    │   ├── lower_bound.py
    │   ├── makeHistogram.py
    │   ├── palettes.py
    │   ├── projplot.py
    │   ├── report.py
    │   ├── sixpanel.py
    │   ├── skew.py
    │   └── thesis_stats/
    │       ├── ideal/
    │       └── amb/
    │
    ├── clouds/
    │   ├── ideal/
    │   └── amb/
    │
    └── arepo_data/
        ├── Kedron_pLoss.npz
        ├── cross_pH2_rel_1e18.npz
        ├── ideal_mhd/
        └── ambipolar_diffusion/
