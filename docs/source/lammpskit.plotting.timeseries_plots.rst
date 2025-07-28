lammpskit.plotting.timeseries\_plots module
============================================

Centralized plotting functions for time series analysis. This module provides reusable plotting functions for creating standardized time series and dual-axis plots with consistent styling and configuration management.

Key Functions
-------------

The module provides functions for:

- **Time series plot creation** with publication-ready defaults
- **Dual-axis plotting** for correlating different physical quantities
- **Figure management** with save and close functionality
- **Statistical labeling** for data summary integration
- **Configuration classes** for consistent plot styling

Configuration System
--------------------

The module uses dataclasses for centralized plot configuration:

**TimeSeriesPlotConfig**
   Controls single-axis time series plots with options for line/scatter combinations, transparency, and typography.

**DualAxisPlotConfig**  
   Extends time series configuration for dual-axis plots with independent color schemes and legend positioning.

Default Configuration Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Standard time series plot defaults
   TimeSeriesPlotConfig(
       alpha=0.55,           # Semi-transparent for overlay visualization
       linewidth=0.1,        # Thin lines for dense time series
       markersize=5,         # Readable markers without overcrowding
       marker='^',           # Triangle markers for distinctiveness
       include_line=True,    # Show trend lines
       include_scatter=True, # Show data points
       format='pdf'          # Vector output for publications
   )

Usage Examples
--------------

Basic time series plot:

.. code-block:: python

   import numpy as np
   from lammpskit.plotting import create_time_series_plot, TimeSeriesPlotConfig
   
   # Generate temporal data
   time = np.linspace(0, 100, 200)  # Time in picoseconds
   connectivity = np.random.rand(200) * 100  # Connectivity percentage
   
   # Create standardized plot
   fig, ax = create_time_series_plot(
       x_data=time,
       y_data=connectivity,
       title='Filament Connectivity Evolution',
       xlabel='Time (ps)',
       ylabel='Connectivity (%)',
       stats_label='Mean: 45.2%, Std: 12.1%'
   )

Customized time series with configuration:

.. code-block:: python

   # Custom configuration for presentation
   config = TimeSeriesPlotConfig(
       alpha=0.8,
       linewidth=0.5,
       markersize=8,
       marker='o',
       format='svg'
   )
   
   fig, ax = create_time_series_plot(
       x_data=time,
       y_data=temperature,
       title='Temperature Evolution',
       xlabel='Time (ps)',
       ylabel='Temperature (K)',
       stats_label='Mean: 315.4K, Std: 25.8K',
       config=config,
       ylim=(280, 350)  # Custom axis limits
   )

Dual-axis comparative analysis:

.. code-block:: python

   from lammpskit.plotting import create_dual_axis_plot, DualAxisPlotConfig
   
   # Configure dual-axis plot
   config = DualAxisPlotConfig(
       primary_color='tab:red',
       secondary_color='tab:blue',
       primary_legend_loc='upper right',
       secondary_legend_loc='lower right'
   )
   
   # Create dual-axis plot for correlation analysis
   fig, ax1, ax2 = create_dual_axis_plot(
       x_data=time,
       primary_y_data=connectivity,
       secondary_y_data=temperature,
       title='Connectivity vs Temperature Evolution',
       xlabel='Time (ps)',
       primary_ylabel='Connectivity (%)',
       secondary_ylabel='Temperature (K)',
       primary_stats_label='Connectivity: 45.2±12.1%',
       secondary_stats_label='Temperature: 315.4±25.8K',
       config=config
   )

Statistical label generation:

.. code-block:: python

   from lammpskit.plotting import calculate_mean_std_label, calculate_frequency_label
   
   # Automatic statistical labeling
   data = np.random.normal(50, 10, 1000)
   stats_label = calculate_mean_std_label(data, 'Connectivity', precision=1)
   print(stats_label)  # "Connectivity: Mean=50.1%, Std=9.8%"
   
   # Frequency analysis labeling
   binary_data = np.random.choice([0, 1], 1000, p=[0.3, 0.7])
   freq_label = calculate_frequency_label(
       binary_data, 1, 'Connected: {frequency:.1f}%', precision=1)
   print(freq_label)  # "Connected: 70.2%"

Font Control Examples
---------------------

Individual font size control:

.. code-block:: python

   fig, ax = create_time_series_plot(
       x_data=time, y_data=data,
       title='Custom Typography',
       xlabel='Time', ylabel='Value',
       stats_label='Statistics',
       fontsize_title=12,    # Larger title
       fontsize_labels=10,   # Medium labels
       fontsize_ticks=8,     # Small tick labels
       fontsize_legend=9     # Medium legend
   )

Configuration-based font control:

.. code-block:: python

   config = TimeSeriesPlotConfig(
       fontsize_title=14,
       fontsize_labels=12,
       fontsize_ticks=10,
       fontsize_legend=11
   )

Performance Considerations
--------------------------

**Rendering Performance**:
   - Transparent plots (alpha < 1.0) may slow rendering for large datasets
   - Vector formats maintain quality but increase file size
   - Line+scatter combination doubles rendering operations

**Memory Management**:
   - Use :func:`save_and_close_figure` for automatic memory cleanup
   - Figure objects are returned for additional customization before saving
   - Large time series (>10⁴ points) benefit from reduced marker density

**Optimization Strategies**:
   - Set ``include_line=False`` for pure scatter analysis
   - Set ``include_scatter=False`` for smooth trend visualization
   - Use ``alpha=1.0`` for faster rendering of simple plots

Common Applications
-------------------

**Filament Evolution Analysis**:
   - Connectivity state over time
   - Gap distance evolution
   - Cluster size distribution temporal changes

**Electrochemical Device Analysis**:
   - Current vs. voltage temporal relationships
   - Temperature vs. resistance correlations
   - Ion mobility temporal patterns

**General MD Analysis**:
   - Property evolution during equilibration
   - Temperature/pressure control validation
   - Energy conservation monitoring

Related Functions
-----------------

- :func:`lammpskit.plotting.plot_multiple_cases` - General comparative plotting
- :func:`lammpskit.ecellmodel.filament_layer_analysis.track_filament_evolution` - Main application
- :func:`lammpskit.ecellmodel.filament_layer_analysis.plot_displacement_timeseries` - Displacement-specific application

Module Documentation
--------------------

.. automodule:: lammpskit.plotting.timeseries_plots
   :members:
   :show-inheritance:
   :undoc-members:
