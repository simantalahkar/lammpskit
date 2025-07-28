API Reference
=============

This section provides comprehensive documentation for all LAMMPSKit modules, functions, and classes.

Package Overview
----------------

.. automodule:: lammpskit
   :members:
   :undoc-members:
   :show-inheritance:

Core Modules
------------

I/O Operations
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   lammpskit.io

The ``lammpskit.io`` module provides functions for reading LAMMPS trajectory files and extracting simulation metadata.

**Key Functions:**

- :func:`lammpskit.io.read_structure_info` - Extract timestep, atom count, and box dimensions
- :func:`lammpskit.io.read_coordinates` - Load atomic coordinates from multiple trajectory files

Plotting Utilities  
~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   lammpskit.plotting

The ``lammpskit.plotting`` module provides general-purpose plotting functions and specialized time series visualization.

**Key Functions:**

- :func:`lammpskit.plotting.plot_multiple_cases` - General comparative plotting utility
- :func:`lammpskit.plotting.create_time_series_plot` - Standardized time series plots
- :func:`lammpskit.plotting.create_dual_axis_plot` - Dual-axis comparative plots

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   lammpskit.config

The ``lammpskit.config`` module provides validation functions and constants for robust LAMMPS analysis.

**Key Functions:**

- :func:`lammpskit.config.validate_file_list` - Validate trajectory file lists
- :func:`lammpskit.config.validate_cluster_parameters` - Validate geometric parameters
- :func:`lammpskit.config.validate_loop_parameters` - Validate timestep ranges

**Constants:**

- ``DEFAULT_COLUMNS_TO_READ`` - Core LAMMPS dump file columns
- ``EXTENDED_COLUMNS_TO_READ`` - Extended column set for comprehensive analysis
- ``DISPLACEMENT_DATA_LABELS`` - Labels for displacement analysis types

Electrochemical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   lammpskit.ecellmodel

The ``lammpskit.ecellmodel`` module provides specialized analysis for electrochemical cell simulations.

**Key Functions:**

- :func:`lammpskit.ecellmodel.filament_layer_analysis.analyze_clusters` - OVITO-based cluster analysis
- :func:`lammpskit.ecellmodel.filament_layer_analysis.track_filament_evolution` - Filament connectivity tracking
- :func:`lammpskit.ecellmodel.filament_layer_analysis.plot_atomic_distribution` - Atomic distribution analysis
- :func:`lammpskit.ecellmodel.filament_layer_analysis.plot_atomic_charge_distribution` - Charge distribution analysis
- :func:`lammpskit.ecellmodel.filament_layer_analysis.plot_displacement_comparison` - Displacement analysis
- :func:`lammpskit.ecellmodel.filament_layer_analysis.plot_displacement_timeseries` - Time series displacement plots

Function Categories
-------------------

Data I/O Functions
~~~~~~~~~~~~~~~~~~

Functions for reading and parsing LAMMPS simulation output:

.. autosummary::
   :toctree: generated/

   lammpskit.io.read_structure_info
   lammpskit.io.read_coordinates
   lammpskit.ecellmodel.read_displacement_data

Analysis Functions
~~~~~~~~~~~~~~~~~~

Functions for scientific analysis of simulation data:

.. autosummary::
   :toctree: generated/

   lammpskit.ecellmodel.analyze_clusters
   lammpskit.ecellmodel.track_filament_evolution
   lammpskit.ecellmodel.plot_atomic_distribution
   lammpskit.ecellmodel.plot_atomic_charge_distribution
   lammpskit.ecellmodel.plot_displacement_comparison

Plotting Functions
~~~~~~~~~~~~~~~~~~

Functions for creating scientific visualizations:

- :func:`lammpskit.plotting.plot_multiple_cases` - Create multi-case comparative plots
- :func:`lammpskit.plotting.create_time_series_plot` - Create standardized time series plots
- :func:`lammpskit.plotting.create_dual_axis_plot` - Create dual-axis correlation plots
- :func:`lammpskit.plotting.save_and_close_figure` - Save and cleanup figure objects
- :func:`lammpskit.ecellmodel.filament_layer_analysis.plot_displacement_timeseries` - Plot temporal displacement analysis

Validation Functions
~~~~~~~~~~~~~~~~~~~~

Functions for input validation and parameter checking:

- :func:`lammpskit.config.validate_filepath` - Validate file paths and existence
- :func:`lammpskit.config.validate_file_list` - Validate lists of file paths
- :func:`lammpskit.config.validate_dataindex` - Validate data index parameters
- :func:`lammpskit.config.validate_loop_parameters` - Validate loop iteration parameters
- :func:`lammpskit.config.validate_chunks_parameter` - Validate chunk size parameters
- :func:`lammpskit.config.validate_cluster_parameters` - Validate clustering analysis parameters

Configuration Classes
~~~~~~~~~~~~~~~~~~~~

Configuration classes for plotting and analysis:

.. autosummary::
   :toctree: generated/

   lammpskit.plotting.TimeSeriesPlotConfig
   lammpskit.plotting.DualAxisPlotConfig
