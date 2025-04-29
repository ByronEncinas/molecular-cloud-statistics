Sure! Here's an updated `README.md` that incorporates the new features of the argument parsing in both `los_stats.py` and `stats.py`.

---

# Simulation Configuration for Cosmic Ray and Plasma Research

This repository provides scripts for running simulations related to cosmic rays and plasma behavior. The scripts allow users to easily configure simulation parameters via command-line arguments. Below is an overview of how to use the scripts and customize their settings.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Command-Line Arguments](#command-line-arguments)
- [Example Usage](#example-usage)
- [Default Values](#default-values)

---

## Requirements

- Python 3.x
- Required Python packages:
  - `argparse`

You can install the required dependencies using `pip`:

```bash
pip install argparse
```

---

## Usage

To run the simulations, you can execute the scripts directly from the command line. The scripts allow you to specify a variety of configuration options, which will be parsed from command-line arguments.

### `los_stats.py`

This script simulates and processes statistics for line-of-sight (LOS) calculations. You can configure the simulation parameters using the following options:

```bash
python los_stats.py [OPTIONS]
```

### `stats.py`

This script calculates statistical properties from the simulation and handles file I/O operations. Similar to `los_stats.py`, this script supports the following configuration options:

```bash
python stats.py [OPTIONS]
```

---

## Command-Line Arguments

### `los_stats.py` Arguments

- `--N` (int): The number of iterations. Default is `4000`.
- `--case` (str): The simulation case type. Can be either `ideal` or `amb`. Default is `ideal`.
- `--num_file` (str): The file number. Default is `'430'`.
- `--max_cycles` (int): The maximum number of cycles. Default is `100`.
- `--NeffOrStability` (str): A flag for calculating stability (`S`) or column densities (`N`). Default is `S`.

#### Example Usage:
```bash
python los_stats.py --N 5000 --case ideal --num_file 500 --max_cycles 200 --NeffOrStability S
```

---

### `stats.py` Arguments

- `--N` (int): The number of iterations. Default is `5000`.
- `--rloc` (float): The reference location. Default is `0.5`.
- `--max_cycles` (int): The maximum number of cycles. Default is `50`.
- `--case` (str): The simulation case type. Can be either `ideal` or `amb`. Default is `ideal`.
- `--num_file` (str): The file number. Default is `'430'`.
- `--seed` (int): The random seed. Default is `12345`.

If less than 6 arguments are provided, `NO_ID` will be appended automatically to the arguments.

#### Example Usage:
```bash
python3 stats.py --N 5000 --rloc 0.7 --max_cycles 100 --case amb --num_file 430 --seed 67890
```

---

## Example Usage

To run the simulations with the default values:

```bash
python3 los_stats.py
```

This will use the default parameters:
- `N = 4000`
- `case = ideal`
- `num_file = 430`
- `max_cycles = 100`
- `NeffOrStability = S`

For `stats.py`, you can run the script similarly with:

```bash
python stats.py
```

This will use the following default values:
- `N = 5000`
- `rloc = 0.5`
- `max_cycles = 50`
- `case = ideal`
- `num_file = 430`
- `seed = 12345`

---

## Default Values

### `los_stats.py`
- **N**: 4000
- **case**: 'ideal'
- **num_file**: '430'
- **max_cycles**: 100
- **NeffOrStability**: 'S'

### `stats.py`
- **N**: 5000
- **rloc**: 0.5
- **max_cycles**: 50
- **case**: 'ideal'
- **num_file**: '430'
- **seed**: 12345

These defaults are used when the respective argument is not provided by the user.

---
