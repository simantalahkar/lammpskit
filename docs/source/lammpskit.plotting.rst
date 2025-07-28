lammpskit.plotting package
==========================

Scientific visualization utilities for creating publication-ready figures across different analysis workflows. The plotting package provides both general-purpose plotting functions and specialized time-series visualization tools with consistent styling and configuration management.

Key Features
------------

- **Publication-ready styling** with scientific color schemes and typography
- **Multi-dimensional array handling** for comparative analysis
- **Centralized configuration** for consistent visual output
- **Time-series specialization** with dual-axis plotting capabilities
- **Memory-efficient rendering** optimized for large datasets
- **Multiple output formats** (PDF, SVG, PNG, EPS) for different use cases

Package Architecture
--------------------

The plotting package is organized into specialized modules:

**General Plotting** (:mod:`~lammpskit.plotting.utils`)
   Flexible plotting functions for scientific data visualization

**Time Series Analysis** (:mod:`~lammpskit.plotting.timeseries_plots`)
   Specialized functions for temporal data analysis with standardized configurations

Core Functions
--------------

General Purpose Plotting:

- :func:`lammpskit.plotting.plot_multiple_cases` - Create multi-case comparative plots

Time Series Plotting:

- :func:`lammpskit.plotting.create_time_series_plot` - Create standardized time series plots
- :func:`lammpskit.plotting.create_dual_axis_plot` - Create dual-axis correlation plots
- :func:`lammpskit.plotting.save_and_close_figure` - Save and cleanup figure objects
- :func:`lammpskit.plotting.calculate_mean_std_label` - Generate statistical labels
- :func:`lammpskit.plotting.calculate_frequency_label` - Generate frequency labels

Configuration Classes:

- :class:`lammpskit.plotting.TimeSeriesPlotConfig` - Configuration for time series plots
- :class:`lammpskit.plotting.DualAxisPlotConfig` - Configuration for dual-axis plots

Styling Standards
-----------------

**Color Palette**: ['b', 'r', 'g', 'k'] (blue, red, green, black)

**Line Styles**: ['--', '-.', ':', '-'] (dashed, dash-dot, dotted, solid)  

**Markers**: ['o', '^', 's', '*'] (circle, triangle, square, star)

**Typography**: 8pt labels, 7pt legends, 7pt ticks for compact scientific layout

Usage Examples
--------------

Basic comparative analysis:

.. code-block:: python

   import numpy as np
   from lammpskit.plotting import plot_multiple_cases
   
   x = np.linspace(0, 30, 50)  # z-positions
   y = np.array([[5, 10, 15], [8, 12, 18]])  # Two cases
   labels = ['SET state', 'RESET state']
   
   fig = plot_multiple_cases(x, y, labels, 'Atom count', 'Z position (Å)', 
                            'comparison', 8, 6)

Time series with dual axes:

.. code-block:: python

   from lammpskit.plotting import create_dual_axis_plot, DualAxisPlotConfig
   
   config = DualAxisPlotConfig(primary_color='tab:red', secondary_color='tab:blue')
   fig, ax1, ax2 = create_dual_axis_plot(
       time, connectivity, temperature, 'Evolution Analysis',
       'Time (ps)', 'Connectivity (%)', 'Temperature (K)',
       'Conn: 45±12%', 'Temp: 315±26K', config=config)

Performance Notes
-----------------

- **Memory usage**: Scales with data size and number of cases
- **Rendering time**: O(n_cases × n_points) for plot generation
- **File I/O**: Vector formats (PDF/SVG) recommended for publications
- **Large datasets**: Consider downsampling for >10⁵ points

Related Modules
---------------

- :mod:`lammpskit.config` - Configuration constants and validation
- :mod:`lammpskit.ecellmodel.filament_layer_analysis` - Analysis functions using these plotting utilities

Submodules
----------

.. toctree::
   :maxdepth: 4

   lammpskit.plotting.timeseries_plots
   lammpskit.plotting.utils

Module contents
---------------

.. automodule:: lammpskit.plotting
   :members:
   :show-inheritance:
   :undoc-members:
   :no-index:
