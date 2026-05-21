
 Reading: arepo_data/ideal_mhd/snap_ideal_000.hdf5

 HEADER attributes:
{'BoxSize': 256.0,
 'Composition_vector_length': 0,
 'Flag_Cooling': 0,
 'Flag_DoublePrecision': 1,
 'Flag_Feedback': 0,
 'Flag_Metals': 0,
 'Flag_Sfr': 0,
 'Flag_StellarAge': 0,
 'Git_commit': b'unknown',
 'Git_date': b'unknown',
 'HubbleParam': 1.0,
 'MassTable': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 'NumFilesPerSnapshot': 1,
 'NumPart_ThisFile': [2445615, 0, 0, 0, 0, 0],
 'NumPart_Total': [2445615, 0, 0, 0, 0, 0],
 'NumPart_Total_HighWord': [0, 0, 0, 0, 0, 0],
 'Omega0': 0.0,
 'OmegaBaryon': 0.0,
 'OmegaLambda': 0.0,
 'Redshift': 0.0,
 'Time': 0.0,
 'UnitLength_in_cm': 3.086e+18,
 'UnitMass_in_g': 1.99e+33,
 'UnitVelocity_in_cm_per_s': 100000.0}

 PARAMETERS (AREPO config options):
{'ActivePartFracForNewDomainDecomp': np.float64(0.0001),
 'AdaptiveHydroSofteningSpacing': np.float64(1.25),
 'BoxSize': np.float64(256.0),
 'CellMaxAngleFactor': np.float64(2.25),
 'CellShapingSpeed': np.float64(0.5),
 'ComovingIntegrationOn': np.int32(0),
 'CoolingGamma': np.float64(2e-26),
 'CoolingOn': np.int32(0),
 'CourantFac': np.float64(0.3),
 'CpuTimeBetRestartFile': np.float64(10800.0),
 'DerefinementCriterion': np.int32(1),
 'DesNumNgb': np.int32(64),
 'ErrTolForceAcc': np.float64(0.0025),
 'ErrTolIntAccuracy': np.float64(0.012),
 'ErrTolTheta': np.float64(0.7),
 'GasSoftFactor': np.float64(1.5),
 'GravityConstantInternal': np.float64(0.0),
 'HubbleParam': np.float64(1.0),
 'ICFormat': np.int32(3),
 'InitCondFile': './IC_test_mhd',
 'InitGasTemp': np.float64(4500.0),
 'JeansNumber': np.float64(32.0),
 'LimitUBelowCertainDensityToThisValue': np.float64(0.0),
 'LimitUBelowThisDensity': np.float64(0.0),
 'MaxMemSize': np.int32(2000),
 'MaxNumNgbDeviation': np.float64(2.0),
 'MaxSizeTimestep': np.float64(0.1),
 'MaxTimebinSpread': np.int32(8),
 'MaxVolume': np.float64(128.0),
 'MaxVolumeDiff': np.float64(8.0),
 'MinEgySpec': np.float64(0.0),
 'MinGasTemp': np.float64(10.0),
 'MinSizeTimestep': np.float64(1e-20),
 'MinVolume': np.float64(0.0001),
 'MinimumComovingHydroSoftening': np.float64(4.5e-10),
 'MinimumDensityOnStartUp': np.float64(0.0),
 'MinimumTargetMass': np.float64(1e-06),
 'MultipleDomains': np.int32(8),
 'NumFilesPerSnapshot': np.int32(1),
 'NumFilesWrittenInParallel': np.int32(32),
 'Omega0': np.float64(0.0),
 'OmegaBaryon': np.float64(0.0),
 'OmegaLambda': np.float64(0.0),
 'OutputDir': './output_actual/',
 'OutputListFilename': 'ol',
 'OutputListOn': np.int32(0),
 'PeriodicBoundariesOn': np.int32(1),
 'ReferenceGasPartMass': np.float64(0.275),
 'RefinementCriterion': np.int32(1),
 'ResubmitCommand': 'my-scriptfile',
 'ResubmitOn': np.int32(0),
 'SnapFormat': np.int32(3),
 'SnapshotFileBase': 'snap',
 'SofteningComovingType0': np.float64(0.1),
 'SofteningComovingType1': np.float64(0.1),
 'SofteningComovingType2': np.float64(0.1),
 'SofteningComovingType3': np.float64(0.1),
 'SofteningComovingType4': np.float64(0.1),
 'SofteningComovingType5': np.float64(0.1),
 'SofteningMaxPhysType0': np.float64(0.1),
 'SofteningMaxPhysType1': np.float64(0.1),
 'SofteningMaxPhysType2': np.float64(0.1),
 'SofteningMaxPhysType3': np.float64(0.1),
 'SofteningMaxPhysType4': np.float64(0.1),
 'SofteningMaxPhysType5': np.float64(0.1),
 'SofteningTypeOfPartType0': np.int32(0),
 'SofteningTypeOfPartType1': np.int32(1),
 'SofteningTypeOfPartType2': np.int32(1),
 'SofteningTypeOfPartType3': np.int32(1),
 'SofteningTypeOfPartType4': np.int32(1),
 'SofteningTypeOfPartType5': np.int32(1),
 'StarformationOn': np.int32(0),
 'TargetGasMassFactor': np.float64(1.0),
 'TimeBegin': np.float64(0.0),
 'TimeBetSnapshot': np.float64(5e-07),
 'TimeBetStatistics': np.float64(0.005),
 'TimeLimitCPU': np.float64(100000.0),
 'TimeMax': np.float64(12.8),
 'TimeOfFirstSnapshot': np.float64(0.0),
 'TopNodeFactor': np.float64(5.0),
 'TypeOfOpeningCriterion': np.int32(1),
 'TypeOfTimestepCriterion': np.int32(0),
 'UnitLength_in_cm': np.float64(3.086e+18),
 'UnitMass_in_g': np.float64(1.99e+33),
 'UnitVelocity_in_cm_per_s': np.float64(100000.0),
 'WaitingTimeFactor': np.float64(1.0)}

🔧 CONFIG (compile-time flags / physics modules):
{'ADAPTIVE_HYDRO_SOFTENING': '',
 'CELL_CENTER_GRAVITY': '',
 'CHUNKING': '',
 'DEBUG': '',
 'DELAY_REFINEMENT': '',
 'DOUBLEPRECISION': np.float64(1.0),
 'ENLARGE_DYNAMIC_RANGE_IN_TIME': '',
 'EVALPOTENTIAL': '',
 'HAVE_HDF5': '',
 'HIERARCHICAL_GRAVITY': '',
 'INPUT_IN_DOUBLEPRECISION': '',
 'ISM_COOLING': '',
 'ISM_COOLING_WITH_BAROTROPIC_EOS': '',
 'KOYAMA_INUTSUKA': '',
 'LOCAL_REFINEMENT': '',
 'LONGIDS': '',
 'MAX_TIMEBIN_SPREAD': '',
 'MAX_TIMEBIN_SPREAD_IN_DOMAIN_DECOMPOSITION': '',
 'MEMORY_MANAGER_CHECK_LEAKS': '',
 'MEMORY_MANAGER_USE_MPROTECT': '',
 'MHD': '',
 'MHD_POWELL': '',
 'NSOFTTYPES_HYDRO': np.float64(128.0),
 'OUTPUT_CENTER_OF_MASS': '',
 'OUTPUT_IN_DOUBLEPRECISION': '',
 'OUTPUT_PRESSURE': '',
 'OVERRIDE_PEANOGRID_WARNING': '',
 'REFINEMENT': '',
 'REFINEMENT_MERGE_CELLS': '',
 'REFINEMENT_SPLIT_CELLS': '',
 'REFINEMENT_VOLUME_LIMIT': '',
 'REGULARIZE_MESH_CM_DRIFT': '',
 'REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED': '',
 'REGULARIZE_MESH_FACE_ANGLE': '',
 'RIEMANN_HLLD': '',
 'SELFGRAVITY': '',
 'TETRA_INDEX_IN_FACE': '',
 'TRACER_FIELD': '',
 'TREE_BASED_TIMESTEPS': '',
 'TURBULENT_BOX': '',
 'VOLUME_LIMIT_ONLY_WITHOUT_TRACER': '',
 'VORONOI': '',
 'VORONOI_DYNAMIC_UPDATE': '',
 'VORONOI_MESH_KEEP_DT_AND_DTC': ''}
🔹 Snapshot file: arepo_data/ideal_mhd/snap_ideal_000.hdf5

Top-level groups:
  Config
  Header
  Parameters
  PartType0

 Header attributes:
  BoxSize: 256.0
  Composition_vector_length: 0
  Flag_Cooling: 0
  Flag_DoublePrecision: 1
  Flag_Feedback: 0
  Flag_Metals: 0
  Flag_Sfr: 0
  Flag_StellarAge: 0
  Git_commit: b'unknown'
  Git_date: b'unknown'
  HubbleParam: 1.0
  MassTable: [0. 0. 0. 0. 0. 0.]
  NumFilesPerSnapshot: 1
  NumPart_ThisFile: [2445615       0       0       0       0       0]
  NumPart_Total: [2445615       0       0       0       0       0]
  NumPart_Total_HighWord: [0 0 0 0 0 0]
  Omega0: 0.0
  OmegaBaryon: 0.0
  OmegaLambda: 0.0
  Redshift: 0.0
  Time: 0.0
  UnitLength_in_cm: 3.086e+18
  UnitMass_in_g: 1.99e+33
  UnitVelocity_in_cm_per_s: 100000.0

 Units: Not found in file

 Particle Types and Fields:
  PartType0:
    - BarotropicEOS: shape=(2445615,), dtype=uint32
    - CenterOfMass: shape=(2445615, 3), dtype=float64
    - Coordinates: shape=(2445615, 3), dtype=float64
    - Density: shape=(2445615,), dtype=float64
    - InternalEnergy: shape=(2445615,), dtype=float64
    - MagneticField: shape=(2445615, 3), dtype=float64
    - MagneticFieldDivergence: shape=(2445615,), dtype=float64
    - MagneticFieldDivergenceAlternative: shape=(2445615,), dtype=float64
    - Masses: shape=(2445615,), dtype=float64
    - ParticleIDs: shape=(2445615,), dtype=uint64
    - Pressure: shape=(2445615,), dtype=float64
    - TracerField: shape=(2445615,), dtype=float64
    - Velocities: shape=(2445615, 3), dtype=float64

 Additional group: Config

 Additional group: Parameters
