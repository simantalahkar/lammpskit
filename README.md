# LAMMPSKit

[![Tests](https://github.com/simantalahkar/lammpskit/actions/workflows/docker-ci.yml/badge.svg)](https://github.com/simantalahkar/lammpskit/actions)
[![Documentation](https://readthedocs.org/projects/lammpskit/badge/?version=latest)](https://lammpskit.readthedocs.io/en/latest/?badge=latest)
[![Coverage](https://codecov.io/gh/simantalahkar/lammpskit/branch/main/graph/badge.svg)](https://codecov.io/gh/simantalahkar/lammpskit)
[![PyPI version](https://badge.fury.io/py/lammpskit.svg)](https://badge.fury.io/py/lammpskit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

LAMMPSKit is a Python toolkit for post-processing and analysing molecular-dynamics (MD) simulations run with [LAMMPS](https://lammps.org/). It was built for simulations of HfTaO-based resistive random-access memory (ReRAM) devices: it tracks how conductive filaments form and evolve, analyses atomic, charge and displacement distributions, and produces publication-ready plots.

## Features

- **Filament analysis**: OVITO-based cluster analysis that tracks filament connectivity, gap, size and height over time
- **Distributions**: atomic, charge and displacement distributions by element and position
- **Large trajectories**: reads only the columns it needs, in batches; tested on trajectories with more than 1M atoms and more than 10K frames
- **Plotting**: publication-ready figures with central control of fonts and styles
- **Modular design**: separate I/O, analysis and plotting modules, with central input validation
- **Tests**: 221 tests, including visual regression tests against baseline images; 96% code coverage

## Installation

```bash
pip install lammpskit
```

Requirements: Python 3.12+, NumPy ≥2.3.1, Matplotlib ≥3.10.3 and OVITO ≥3.12.4.

For development:

```bash
git clone https://github.com/simantalahkar/lammpskit.git
cd lammpskit
pip install -e .[dev]
# or, if that fails: pip install -e . && pip install -r requirements-dev.txt
```

The development extras add pytest, pytest-cov and pytest-mpl for testing, Sphinx for the documentation, black, flake8, isort and mypy for code quality, and build and twine for packaging.

## Function summary

### I/O (`lammpskit.io`)
| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| read_structure_info            | Read trajectory metadata (timestep, atom count, box dimensions) |
| read_coordinates               | Load coordinates and cell information from trajectory files   |

### Analysis (`lammpskit.ecellmodel.filament_layer_analysis`)
| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| analyze_clusters               | OVITO-based cluster analysis and filament properties          |
| track_filament_evolution       | Track filament connectivity, gap and size over time           |
| read_displacement_data         | Read processed displacement data files                        |
| plot_atomic_distribution       | Analyse and plot atomic distributions by element              |
| plot_atomic_charge_distribution| Analyse and plot atomic charge distributions                  |
| plot_displacement_comparison   | Compare displacement data across cases                        |
| plot_displacement_timeseries   | Plot displacement time series                                 |

### Data processing (`lammpskit.ecellmodel.data_processing`)
| Function                             | Purpose                                                  |
|--------------------------------------|----------------------------------------------------------|
| select_atom_types_from_coordinates   | Select atoms by type from coordinate arrays              |
| calculate_z_bins_setup               | Set up spatial bins along z                              |
| calculate_atomic_distributions       | Atomic distributions per bin                             |
| calculate_charge_distributions       | Charge distributions per bin                             |
| extract_element_label_from_filename  | Read the element label from a file name                  |

### Plotting (`lammpskit.plotting`)
| Function                       | Purpose                                                        |
|--------------------------------|----------------------------------------------------------------|
| plot_multiple_cases            | General plotting utility for several cases                    |
| timeseries_plots.*             | Time-series plots with central font and style control         |

Input validation functions live in `lammpskit.config`.

## Example workflow

`usage/ecellmodel/run_analysis.py` shows four analyses:

1. **Filament evolution**: filament connectivity and structure over time
2. **Displacement**: atomic displacements compared across species (Hf, O, Ta)
3. **Charge distribution**: local charge distributions in the device
4. **Atomic distribution**: atomic distributions under different applied voltages

The simulation data it reads is not included in the repository. Put your LAMMPS trajectory and displacement files in `usage/ecellmodel/data/` (see the paths in the script), then run it from the repository root:

```bash
python usage/ecellmodel/run_analysis.py
```

Plots are written to `usage/ecellmodel/output/`, which holds the figures from the author's own run.

### Package structure

```
lammpskit/
├── io/            # Reading LAMMPS files
├── plotting/      # Plotting utilities and time-series plots
├── ecellmodel/    # Analysis of electrochemical (ReRAM) cell simulations
└── config.py      # Configuration and input validation
```

## Docker image

A Docker image with all dependencies installed is available on [Docker Hub](https://hub.docker.com/r/simantalahkar/lammpskit).

1. Install Docker (see [Get Docker](https://docs.docker.com/get-docker/)).

2. Pull the image:
   ```sh
   docker pull simantalahkar/lammpskit:latest
   # or a specific version
   docker pull simantalahkar/lammpskit:1.2.1
   ```

3. Run it with your data mounted as a volume:
   ```sh
   docker run -it -v /path/to/your/data:/data simantalahkar/lammpskit:latest
   ```
   This starts a bash shell in the container, with your data at `/data`.

4. Use the package:
   ```sh
   python
   >>> import lammpskit
   ```

5. Copy scripts into the container from another terminal, or install more packages inside it:
   ```sh
   docker cp /path/to/local/script.py <container_id>:/home/lammpsuser/
   pip install <package-name>
   ```

6. Leave with `exit`. The container stays on your system: re-enter it with `docker start <container_id>` and `docker exec -it <container_id> bash`, or delete it with `docker rm <container_id>`.

## Tests

The test suite has 221 tests, including visual regression tests against baseline images in `tests/baseline/`. Tests that need OVITO are skipped when it is not installed. Tests are not shipped with the PyPI package.

- `tests/test_io.py`: I/O functions
- `tests/test_plotting.py`: plotting utilities
- `tests/test_timeseries_plots.py`: time-series plots
- `tests/test_centralized_font_control.py`: font and style consistency
- `tests/test_config.py`: configuration and validation
- `tests/test_ecellmodel/`: analysis functions

```sh
pip install -e .[dev]

# Run all tests (with coverage, as set in pyproject.toml)
pytest

# Run with visual regression testing
pytest --mpl

# Regenerate the baseline images after an intended change to a plot
pytest --mpl-generate-path=tests/baseline tests/
```

All baseline images are kept in `tests/baseline/`. Tests in `tests/` use `baseline_dir="baseline"` and tests in `tests/test_ecellmodel/` use `baseline_dir="../baseline"`, so the same paths work locally, in Docker and in CI:

```python
@pytest.mark.mpl_image_compare(baseline_dir="../baseline", remove_text=True)
def test_plotting_function():
    return figure
```

## Data formats

### LAMMPS trajectory file (`.lammpstrj`)
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

### Processed displacement data file
```
# header1
# header2
# header3
0 2
1.0 3.0
2.0 4.0
# end loop
```

## Changelog

See [CHANGELOG.md](https://github.com/simantalahkar/lammpskit/blob/main/CHANGELOG.md).

## Citation

If you use this package in your research, please cite:

S. Lahkar et al., Decoupling Electric Field and Temperature-Driven Atomistic Forming Mechanisms in TaOx/HfO2-Based ReRAMs using Reactive Molecular Dynamics Simulations, arXiv:2505.24468 (2025), https://doi.org/10.48550/arXiv.2505.24468


## License

GPL-3.0-or-later

## Author

Simanta Lahkar

## Links

- [Homepage](https://github.com/simantalahkar/lammpskit)
- [Documentation](https://lammpskit.readthedocs.io/en/latest/)
- [Changelog](./CHANGELOG.md)