lammpskit.plotting.utils module
===============================

General-purpose plotting utilities for scientific visualization. This module provides flexible plotting functions for creating publication-ready figures across different analysis workflows with automatic styling and extensive customization options.

Design Philosophy
-----------------

Functions prioritize **flexibility over rigid interfaces**, using ``**kwargs`` for extensive customization while maintaining consistent visual output across the LAMMPSKit ecosystem.

Core Function
-------------

**plot_multiple_cases** - Create multi-case comparative plots with automatic styling and layout management.

Array Dimension Support
-----------------------

The :func:`plot_multiple_cases` function handles multiple array configurations:

.. list-table:: Dimension Handling
   :header-rows: 1
   :widths: 25 25 50

   * - x_arr dimensions
     - y_arr dimensions  
     - Usage scenario
   * - 1D
     - 1D
     - Single case plot
   * - 1D  
     - 2D
     - Shared x-axis, multiple y-series
   * - 2D
     - 1D
     - Multiple x-series, shared y-axis
   * - 2D
     - 2D
     - Full multi-case plot (most common)

Customization Options
---------------------

**Axis Control**:
   - ``xlimit``, ``ylimit`` - Set both axis limits
   - ``xlimitlo``, ``xlimithi`` - Set individual axis limits
   - ``ylimitlo``, ``ylimithi`` - Set individual axis limits

**Reference Lines**:
   - ``xaxis=True`` - Add horizontal line at y=0
   - ``yaxis=True`` - Add vertical line at x=0

**Styling**:
   - ``markerindex`` - Override automatic color/marker cycling

**Statistical Analysis**:
   - ``ncount`` - Atom counts per bin for weighted average calculations

Usage Examples
--------------

Electrochemical cell analysis:

.. code-block:: python

   import numpy as np
   from lammpskit.plotting import plot_multiple_cases
   
   # Atomic distribution along z-axis (electrode-to-electrode)
   z_positions = np.linspace(-10, 40, 50)  
   hf_distribution = np.array([[10, 15, 20], [5, 12, 18]])
   labels = ['SET state', 'RESET state']
   
   fig = plot_multiple_cases(hf_distribution, z_positions, labels,
                            'Hf atom count', 'Z position (Å)', 
                            'hafnium_analysis', 10, 8)

Displacement analysis with reference lines:

.. code-block:: python

   displacement = np.random.normal(0, 1, 100)
   positions = np.linspace(-10, 40, 100)
   
   fig = plot_multiple_cases(displacement, positions, ['Displacement'],
                            'Displacement (Å)', 'Z position (Å)',
                            'displacement_profile', 8, 6, 
                            yaxis=True, xaxis=True)  # Add reference lines

Multi-case with axis limits:

.. code-block:: python

   charge_data = np.array([[1, 2, 3], [2, 4, 6], [1.5, 3, 4.5]])
   z_pos = np.array([0, 10, 20])
   labels = ['0.5V', '1.0V', '1.5V']
   
   fig = plot_multiple_cases(charge_data, z_pos, labels,
                            'Net charge', 'Z position (Å)', 'charge_dist', 8, 6,
                            ylimithi=70, xlimithi=25, xlimitlo=-5)

Statistical analysis with atom counts:

.. code-block:: python

   distributions = np.array([[10, 15, 20], [8, 12, 16]])
   atom_counts = np.array([[100, 150, 200], [80, 120, 160]])  # For weighted averages
   
   fig = plot_multiple_cases(distributions, z_positions, labels,
                            'Atom density', 'Z position (Å)', 'density', 8, 6,
                            ncount=atom_counts)  # Prints weighted averages

Output Format
-------------

**Dual Format Output**: All plots are automatically saved in both PDF and SVG formats:
   - ``{output_filename}.pdf`` - Vector format for publications
   - ``{output_filename}.svg`` - Web-compatible vector format

**Memory Management**: Figures are automatically closed after saving for memory efficiency

Performance Considerations
--------------------------

- **Memory usage**: O(max(x_size, y_size)) per plot
- **Rendering time**: O(n_cases × n_points) for line/marker rendering
- **Large datasets**: Consider data downsampling for >10⁵ points
- **Batch processing**: Function optimized for multiple figure generation

Common Applications
-------------------

**Electrochemical Analysis**:
   - Atomic distributions under different voltages
   - Charge distributions across electrode separation
   - Ion migration patterns vs. position

**Displacement Studies**:
   - Atomic displacement profiles
   - Temperature-dependent mobility
   - Comparative displacement analysis

**General MD Analysis**:
   - Property distributions vs. spatial coordinates
   - Time-averaged quantities comparison
   - Multi-condition analysis workflows

Related Functions
-----------------

- :func:`lammpskit.plotting.create_time_series_plot` - Specialized time series plotting
- :func:`lammpskit.plotting.create_dual_axis_plot` - Dual-axis comparative plots
- :func:`lammpskit.config.validate_file_list` - Input validation for batch processing

Module Documentation
--------------------

.. automodule:: lammpskit.plotting.utils
   :members:
   :show-inheritance:
   :undoc-members:
