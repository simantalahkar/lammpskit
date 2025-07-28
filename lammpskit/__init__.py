"""
LAMMPSKit: A modular toolkit for LAMMPS molecular dynamics simulation analysis.

LAMMPSKit provides a streamlined framework for analyzing raw data from molecular dynamics simulations,
with particular emphasis on electrochemical systems. The package follows a hierarchical
design allowing both general-purpose trajectory analysis and specialized domain-specific
workflows.

Modules
-------
io : module
    LAMMPS trajectory file readers and data parsers. Handles standard LAMMPS dump formats
    with automatic column detection and efficient memory management for large datasets.

plotting : module
    Scientific visualization utilities with standardized styling and publication-ready
    output. Includes time series analysis, dual-axis plots, and statistical overlays.

config : module
    Centralized configuration management for file paths, analysis parameters, and
    computational settings. Supports both programmatic and file-based configuration.

ecellmodel : module
    Specialized analysis for electrochemical cell simulations. Provides filament tracking,
    charge distribution and displacement analyses, and clustering algorithms optimized for HfTaO systems.

Target Applications
-------------------
- Electrochemical memory device simulations (ReRAM, memristors).
- Ion transport and defect migration studies.
- Phase transition analysis in oxide materials.
- General LAMMPS trajectory post-processing.

Performance Notes
-----------------
Memory usage scales linearly with trajectory size. For datasets >1GB, use chunked
reading via io.lammps_readers with appropriate buffer sizes.

Community Extensions
--------------------
To extend for other MD simulation types:
1. Add new analysis modules following ecellmodel structure
2. Implement domain-specific atom type mappings in config.
3. Create specialized plotting functions for new data types.
4. Add validation functions for new simulation parameters.

Examples
--------
Basic workflow for electrochemical analysis:

>>> import lammpskit as lk
>>> from lammpskit.config import DEFAULT_COLUMNS_TO_READ
>>> file_list = ['trajectory1.lammpstrj', 'trajectory2.lammpstrj']
>>> coords, timesteps, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = lk.io.read_coordinates(
...     file_list, skip_rows=9, columns_to_read=DEFAULT_COLUMNS_TO_READ)
>>> from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters
>>> analyze_clusters('trajectory1.lammpstrj')

General trajectory parsing and plotting:

>>> timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = lk.io.read_structure_info('dump.lammpstrj')
>>> import numpy as np
>>> x_data, y_data = np.array([1, 2, 3]), np.array([[4, 5, 6]])
>>> lk.plotting.plot_multiple_cases(x_data, y_data, ['Case1'], 'X', 'Y', 'output', 8, 6)
"""

# Version info
__version__ = "1.1.0"

# Import key modules for convenient access
from . import io
from . import plotting
from . import config
from . import ecellmodel

__all__ = [
    "io",
    "plotting",
    "config",
    "ecellmodel",
]
