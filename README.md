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
        ├── images/*/*.png
        │
        ├── gists/
        │   ├── crutcher.py
        │   ├── divergence.py
        │   ├── gabriel.py
        │   ├── los_stats.py
        │   ├── palettes.py
        │   └── skew.py
        │
        ├── clouds/
        │   ├── ideal/
        │   └── amb/
        │
        └── arepo_data/
            ├── pLoss.npz
            ├── cross_pH2_rel_1e18.npz
            ├── ideal_mhd/
            └── ambipolar_diffusion/
