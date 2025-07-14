# lammpskit

**lammpskit** is a Python toolkit for running, post-processing, and analyzing molecular dynamics (MD) simulations with [LAMMPS](https://lammps.org/). It provides a collection of functions and scripts to streamline simulation workflows and automate analysis of LAMMPS output data. While designed for computational materials science, its modular data processing and analysis functions are broadly applicable to scientific computing, data engineering, and machine learning workflows involving structured time series or atomic-scale data.

## Features

- Utilities for reading and processing LAMMPS trajectory and data files
- Functions for statistical analysis and visualization of simulation results
- Tools for analyzing atomic distributions, charges, displacements, and clusters
- Modular plotting utilities for scientific data
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

**Runtime requirements (for using the package):**
- Python 3.12+
- numpy
- matplotlib
- ovito

**Development & testing requirements (for contributing, testing, or building):**
- All runtime requirements above
- pytest
- pytest-cov
- pytest-mpl
- Jinja2
- MarkupSafe
- pillow
- packaging
- build
- coverage
- cycler
- fonttools
- pluggy
- pyparsing
- python-dateutil
- readme_renderer
- requests
- requests-toolbelt
- rich
- setuptools
- twine


## Installation (PyPI)

For end users (runtime):
```sh
pip install lammpskit
```

For development and testing:
```sh
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Development and Testing Environment

All runtime dependencies are listed in `requirements.txt`. Development and test dependencies are listed in `requirements-dev.txt` and in the `[dev]` group of `pyproject.toml` and `setup.py`.

To set up a development environment and run tests locally:
```sh
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```
Tests are not shipped with the PyPI package, but are available in the source repository for development and validation.

## Data Format Examples

### LAMMPS Trajectory File (`.lammpstrj`)
```
ITEM: TIMESTEP
1200000
ITEM: NUMBER OF ATOMS
5
ITEM: BOX BOUNDS pp pp pp
0.0 50.0
0.0 50.0
0.0 50.0
ITEM: ATOMS id type q x y z ix iy iz vx vy vz c_eng
1 2 0.1 1.0 2.0 3.0 0 0 0 0 0 0 0
2 1 -0.2 2.0 3.0 4.0 0 0 0 0 0 0 0
... (one line per atom)
```

### Processed Displacement Data File
```
# header1
# header2
# header3
0 2
1.0 3.0
2.0 4.0
# end loop
```

## Main Functions

### `read_structure_info(filepath)`
Reads a LAMMPS trajectory file and returns timestep, atom count, and box dimensions. Robust to missing/malformed data.

### `read_coordinates(file_list, skip_rows, columns_to_read)`
Loads atomic coordinates and simulation cell parameters from a list of trajectory files.

### `read_displacement_data(filepath, loop_start, loop_end, repeat_count=0)`
Reads binwise averaged displacement data from processed output files, handling chunked data and errors.

### `plot_multiple_cases(x_arr, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize, ...)`
General plotting utility for 1D/2D arrays, supporting various plot customizations.

### `plot_atomic_distribution(file_list, labels, skip_rows, z_bins, analysis_name, output_dir)`
Reads coordinates, computes atomic distributions (O, Hf, Ta, metals), and plots stoichiometry and atom counts.

### `plot_atomic_charge_distribution(file_list, labels, skip_rows, z_bins, analysis_name, output_dir)`
Computes and plots charge distributions and mean charges for atom types.

### `plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count, output_dir)`
Reads displacement data for multiple cases, plots z/lateral displacements vs. bin positions.

### `analyze_clusters(filepath)`
Uses OVITO to perform cluster analysis, coordination, and connectivity checks on metallic atoms, returning filament properties.

### `track_filament_evolution(file_list, analysis_name, time_step, dump_interval_steps, output_dir)`
Tracks filament connectivity/gap/separation/size over time, and generates summary plots.

### `plot_displacement_timeseries(file_list, datatype, dataindex, Nchunks, loop_start, loop_end, output_dir)`
Plots time series of displacement data for selected data indices.

### `run_analysis(...)`
Orchestrates all analysis workflows, setting up parameters, file lists, and calling the above functions for various scenarios.

## Function Summary Table

| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| read_structure_info            | Parse trajectory metadata                                      |
| read_coordinates               | Load coordinates and cell info                                 |
| read_displacement_data         | Parse processed displacement data                              |
| plot_multiple_cases            | General scientific plotting utility                            |
| plot_atomic_distribution       | Analyze and plot atomic distributions                          |
| plot_atomic_charge_distribution| Analyze and plot atomic charge distributions                   |
| plot_displacement_comparison   | Compare displacement data across cases                         |
| analyze_clusters               | Cluster analysis and filament property extraction              |
| track_filament_evolution       | Track filament evolution over time                             |
| plot_displacement_timeseries   | Plot time series of displacement data                          |
| run_analysis                   | Main workflow orchestration                                    |

## Test Coverage

Extensive test coverage is provided for all major functions, including edge cases and typical usage scenarios. Tests are located in the `tests/test_ecellmodel` folder in the GitHub repository and use `pytest` and `pytest-mpl` for automated validation and image comparison.

To run tests locally:
```sh
pip install .[dev]
pytest
```
Tests are not shipped with the PyPI package, but are available in the source repository for development and validation.


## Docker Image

An official Docker image for **lammpskit** is available on [Docker Hub](https://hub.docker.com/r/simantalahkar/lammpskit).
Using the Docker container provides a portable, reproducible environment for running lammpskit anywhere Docker is supported.

### Benefits

- Portable and reproducible environment
- All dependencies pre-installed
- Easy to use on any system with Docker

### How to Use

1. **Install Docker** on your system.  
   See [Get Docker](https://docs.docker.com/get-docker/) for instructions.

2. **Pull the latest image:**
   ```sh
   docker pull simantalahkar/lammpskit:latest
   ```
   Or pull a specific version:
   ```sh
   docker pull simantalahkar/lammpskit:0.2.1
   ```

3. **Run the container with your local data mounted as a volume:**
   ```sh
   docker run -it -v /path/to/your/data:/data simantalahkar/lammpskit:latest
   ```
   This starts a bash shell in the container. Your local data is accessible at `/data`.

4. **Use the installed Python package:**
   ```sh
   python
   >>> import lammpskit
   # ...your analysis code...
   ```

5. **Copy custom scripts or files into the container (from another terminal):**
   ```sh
   docker cp /path/to/local/script.py <container_id>:/home/lammpsuser/
   ```
   You can also install additional Python packages inside the container:
   ```sh
   pip install <package-name>
   ```

6. **Exit the container after analysis:**
   ```sh
   exit
   ```
   The container will remain on your system for future use.  
   To re-enter the container:
   ```sh
   docker start <container_id>
   docker exec -it <container_id> bash
   ```
   To delete the container completely:
   ```sh
   docker rm <container_id>
   ```


## Changelog

See the [CHANGELOG.md](./CHANGELOG.md) for a detailed list of changes and updates.

## Citation

If you use this package in your research, please cite:

S. Lahkar et al., Decoupling Local Electrostatic Potential and Temperature-Driven Atomistic Forming Mechanisms in TaOx/HfO2-Based ReRAMs using Reactive Molecular Dynamics Simulations, arXiv:2505.24468, https://doi.org/10.48550/arXiv.2505.24468


## License

GPL-3.0-or-later

## Author

Simanta Lahkar

## Links

- [Homepage](https://github.com/simantalahkar/lammpskit)
- [Documentation](https://github.com/simantalahkar/lammpskit#readme)
- [Changelog](./CHANGELOG.md)