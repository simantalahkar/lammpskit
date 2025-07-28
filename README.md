# LAMMPSKit

[![Tests](https://github.com/simantalahkar/lammpskit/workflows/Tests%20and%20Quality%20Checks/badge.svg)](https://github.com/simantalahkar/lammpskit/actions)
[![Documentation](https://readthedocs.org/projects/lammpskit/badge/?version=latest)](https://lammpskit.readthedocs.io/en/latest/?badge=latest)
[![Coverage](https://codecov.io/gh/simantalahkar/lammpskit/branch/main/graph/badge.svg)](https://codecov.io/gh/simantalahkar/lammpskit)
[![PyPI version](https://badge.fury.io/py/lammpskit.svg)](https://badge.fury.io/py/lammpskit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**LAMMPSKit** is a comprehensive Python toolkit for molecular dynamics (MD) simulation analysis with [LAMMPS](https://lammps.org/), specialized for electrochemical memory device characterization and filament analysis. It provides advanced post-processing capabilities for HfTaO-based resistive random access memory (ReRAM) devices, including filament formation tracking, charge redistribution analysis, and temporal evolution characterization.

## 🚀 Features

- **🔬 Specialized Analysis**: Advanced filament evolution tracking for electrochemical memory devices
- **📊 Comprehensive Visualization**: Publication-ready scientific plotting with customizable styling
- **⚡ High Performance**: Optimized for large trajectory analysis (>1M atoms, >10K frames)
- **🧪 OVITO Integration**: Advanced cluster analysis and structural characterization
- **📐 Robust Data Processing**: Memory-efficient streaming for multi-gigabyte datasets
- **🎯 Modular Design**: Separate I/O, plotting, and analysis components for flexibility
- **🔧 Configuration Management**: Centralized validation and parameter management
- **✅ Complete Testing**: 96% code coverage with visual regression testing

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install lammpskit
```

### For Development
```bash
git clone https://github.com/simantalahkar/lammpskit.git
cd lammpskit

# Primary method (modern pip with optional dependencies)
pip install -e .[dev]

# Alternative method (if above fails)
pip install -e . && pip install -r requirements-dev.txt
```

## 📋 Requirements

**Runtime Dependencies:**
- Python 3.12+
- numpy ≥2.3.1
- matplotlib ≥3.10.3
- ovito ≥3.12.4

**Development Dependencies:**
- All runtime requirements
- Testing: pytest, pytest-cov, pytest-mpl
- Documentation: sphinx, sphinx-rtd-theme
- Code Quality: black, flake8, isort, mypy
- Build Tools: build, twine, setuptools



## Function Summary Table

### Core I/O Functions (`lammpskit.io.lammps_readers`)
| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| read_structure_info            | Parse trajectory metadata (timestep, atom count, box dims)    |
| read_coordinates               | Load coordinates and cell info from trajectory files          |
| read_displacement_data         | Parse processed displacement data with robust error handling   |

### Analysis Functions (`lammpskit.ecellmodel.filament_layer_analysis`)
| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| analyze_clusters               | OVITO-based cluster analysis and filament property extraction |
| track_filament_evolution       | Track filament connectivity, gap, and size over time          |
| plot_atomic_distribution       | Analyze and plot atomic distributions by element type         |
| plot_atomic_charge_distribution| Analyze and plot atomic charge distributions                   |
| plot_displacement_comparison   | Compare displacement data across multiple cases               |
| plot_displacement_timeseries   | Plot time series of displacement data with customization      |

### Plotting Utilities (`lammpskit.plotting`)
| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| plot_multiple_cases            | General scientific plotting utility with flexible styling     |
| timeseries_plots.*             | Centralized timeseries plotting with font and style control   |

### Data Processing (`lammpskit.ecellmodel.data_processing`)
| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| Various validation functions   | Centralized input validation and error handling               |

### Example Workflows (`usage/ecellmodel/`)
| Script                         | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| run_analysis.py               | Complete workflow demonstrating 4 main analysis types        |


## Quick Start

### Using the Example Workflow

LAMMPSKit v1.1.0 includes a comprehensive example workflow that demonstrates all major package capabilities:

```python
# Clone the repository and navigate to the usage example
git clone https://github.com/simantalahkar/lammpskit.git
cd lammpskit/usage/ecellmodel

# Run the complete analysis workflow
python run_analysis.py
```

This workflow demonstrates four main analysis types:
1. **Filament Evolution Tracking** - Monitor filament connectivity and structural changes over time
2. **Displacement Analysis** - Compare atomic displacements across different species (Hf, O, Ta)
3. **Charge Distribution Analysis** - Analyze local charge distributions in electrochemical systems
4. **Atomic Distribution Analysis** - Study atomic distributions under different applied voltages

Generated plots and analysis results are saved to `usage/ecellmodel/output/`.

### Package Structure

LAMMPSKit v1.1.0 features a modular architecture:

```
lammpskit/
├── io/                    # Data reading and I/O operations
├── plotting/              # Visualization utilities and timeseries plots
├── ecellmodel/           # Electrochemical analysis functions
└── config.py             # Centralized configuration management
```

## Docker Image

An official Docker image for **lammpskit** is available on [Docker Hub](https://hub.docker.com/r/simantalahkar/lammpskit).
Using the Docker container provides a portable, reproducible environment with all dependencies pre-installed for running and testing lammpskit anywhere Docker is supported.

### How to Use

1. **Install Docker** on your system.  
   See [Get Docker](https://docs.docker.com/get-docker/) for instructions.

2. **Pull the latest image:**
   ```sh
   docker pull simantalahkar/lammpskit:latest
   ```
   Or pull a specific version:
   ```sh
   docker pull simantalahkar/lammpskit:1.1.0
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

## Installation (PyPI)

For end users (runtime):
```sh
pip install lammpskit
```

For development and testing:
```sh
# Method 1: Using optional dependencies (recommended)
pip install -e .[dev]

# Method 2: Manual installation (fallback)
pip install -e .
pip install -r requirements-dev.txt
```

## Development and Testing Environment

All runtime dependencies are listed in `requirements.txt`. Development and test dependencies are available in multiple formats:
- `[project.optional-dependencies]` in `pyproject.toml` (modern standard)
- `requirements-dev.txt` (traditional method)  
- `extras_require` in `setup.py` (legacy compatibility)

To set up a development environment and run tests locally:
```sh
# Recommended approach with fallback
pip install -e .[dev] || (pip install -e . && pip install -r requirements-dev.txt)
pytest
```
Tests are not shipped with the PyPI package, but are available in the source repository for development and validation.


## Test Coverage

LAMMPSKit v1.1.0 includes extensive test coverage with over 270 test functions and 205 baseline images for visual regression testing. Tests are organized by module and include edge cases and typical usage scenarios:

- `tests/test_io.py` - I/O function validation
- `tests/test_plotting.py` - General plotting utilities  
- `tests/test_timeseries_plots.py` - Timeseries plotting functions
- `tests/test_ecellmodel/` - Analysis function validation
- `tests/test_config.py` - Configuration management

To run tests locally:
```sh
pip install .[dev]
pytest
```
Tests use `pytest` and `pytest-mpl` for automated validation and image comparison. Tests are not shipped with the PyPI package, but are available in the source repository for development and validation.


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


## Changelog

See the [CHANGELOG.md](https://github.com/simantalahkar/lammpskit/blob/main/CHANGELOG.md) for a detailed list of changes and updates.

## Citation

If you use this package in your research, please cite:

S. Lahkar et al., Decoupling Local Electrostatic Potential and Temperature-Driven Atomistic Forming Mechanisms in TaOx/HfO2-Based ReRAMs using Reactive Molecular Dynamics Simulations, arXiv:2505.24468, https://doi.org/10.48550/arXiv.2505.24468


## License

GPL-3.0-or-later

## Author

Simanta Lahkar

## Links

- [Homepage](https://github.com/simantalahkar/lammpskit)
- [Documentation](https://lammpskit.readthedocs.io/)
- [Changelog](./CHANGELOG.md)