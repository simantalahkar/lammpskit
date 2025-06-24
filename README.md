# lammpskit

**lammpskit** is a Python toolkit for running, post-processing, and analyzing molecular dynamics (MD) simulations with [LAMMPS](https://lammps.org/). It provides a collection of functions and scripts to streamline simulation workflows and automate analysis of LAMMPS output data.

## Features

- Utilities for reading and processing LAMMPS trajectory and data files
- Functions for statistical analysis and visualization of simulation results
- Tools for analyzing atomic distributions, charges, displacements, and clusters
- Example scripts for common analysis tasks

## Installation

```sh
pip install lammpskit
```

Or, for development:

```sh
pip install .[dev]
```

## Requirements

- Python 3.12+
- numpy
- matplotlib
- ovito

## Usage

Import and use the toolkit in your own scripts:

```python
import lammpskit

# Example: analyze a trajectory
lammpskit.plot_atomic_distribution(...)
```

See the [examples](examples/) and function docstrings for more details.

## License

GPL-3.0-or-later

## Author

Simanta Lahkar

## Links

- [Homepage](https://github.com/simantalahkar/lammpskit)
- [Documentation](https://github.com/simantalahkar/lammpskit#readme)