Fix argument parsing for configurable simulation parameters

# library.py

Physical constans

# los_stats.py

Enhance argument parsing for simulation configuration

- Improved command-line argument handling to allow users to specify:
  - N: The number of iterations (default: 4000)
  - case: The simulation case type ('ideal' or 'amb', default: 'ideal')
  - num_file: The file number (default: '430')
  - max_cycles: The maximum number of cycles (default: 100)
  - NeffOrStability: A flag to specify whether to calculate stability ('S') or column densities ('N', default: 'S')
  
- If no command-line arguments are provided, the script will use the default values.
- This change provides greater flexibility for configuring the simulation through command-line arguments.

# stats.py

Refactor argument parsing for simulation parameters and file handling

- Enhanced command-line argument handling to allow users to specify:
  - N: The number of iterations (default: 5000)
  - rloc: The reference location (default: 0.5)
  - max_cycles: The maximum number of cycles (default: 50)
  - case: The simulation case type (e.g., 'ideal', default: 'ideal')
  - num_file: The file number (default: '430')
  
- Added logic to append 'NO_ID' to the arguments if less than 6 arguments are provided.
- Defaults are used if no command-line arguments are given, making it easier to run the script without manual adjustments.
