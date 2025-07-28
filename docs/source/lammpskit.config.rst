lammpskit.config module
=======================

The configuration module provides essential validation functions and constants for robust LAMMPS trajectory analysis. These functions ensure input validation, parameter checking, and standardized constants used across LAMMPSKit analysis workflows.

Key Functions
-------------

The module includes validation functions for:

- **File path validation** with existence checking
- **Data index validation** with range checking  
- **File list validation** for batch processing
- **Loop parameter validation** for iteration control
- **Chunk parameter validation** for parallel processing
- **Cluster parameter validation** for analysis configuration

Plot Configuration System
-------------------------

**PlotConfig (Base Class)**

The foundation for all plotting configurations, providing:

- Typography control (title, labels, ticks, legend fonts)
- Color scheme management
- Figure size and DPI settings
- Output format specifications

Default Typography Settings:
   - Title: 12pt (emphasis without overwhelming)
   - Labels: 10pt (clear readability)
   - Ticks: 8pt (compact but legible)
   - Legend: 9pt (balanced information density)

**ScatterPlotConfig**

Extends PlotConfig for scatter plot specific parameters:

.. code-block:: python

   ScatterPlotConfig(
       alpha=0.5,          # Semi-transparency for overlap visualization
       marker='o',         # Circular markers for universal recognition
       markersize=30,      # Large enough for pattern recognition
       colormap='viridis', # Perceptually uniform colormap
       format='pdf'        # Vector output for publications
   )

**HistogramConfig**

Specialized for histogram and distribution plots:

.. code-block:: python

   HistogramConfig(
       bins=30,            # Balanced between detail and noise
       alpha=0.7,          # Slight transparency for overlays
       density=True,       # Normalized distributions
       color='skyblue',    # Professional, non-aggressive color
       edgecolor='black'   # Clear bin boundaries
   )

**TimeSeriesPlotConfig**

Optimized for temporal data visualization:

.. code-block:: python

   TimeSeriesPlotConfig(
       alpha=0.55,         # Transparency for overlay analysis
       linewidth=0.1,      # Thin lines for dense time series
       markersize=5,       # Readable without overcrowding
       marker='^',         # Distinctive triangular markers
       include_line=True,  # Show trend lines
       include_scatter=True # Show individual data points
   )

Font Configuration Examples
---------------------------

Centralized font control:

.. code-block:: python

   from lammpskit.config import CentralizedFontConfig
   
   # Professional presentation configuration
   presentation_fonts = CentralizedFontConfig(
       title_size=16,
       label_size=14,
       tick_size=12,
       legend_size=13,
       family='serif'  # Traditional academic style
   )
   
   # Apply to any plot configuration
   plot_config = TimeSeriesPlotConfig(font_config=presentation_fonts)

Individual font overrides:

.. code-block:: python

   # Override specific font sizes while keeping others at defaults
   config = PlotConfig(
       fontsize_title=14,    # Larger title only
       fontsize_labels=10,   # Keep default label size
       # Other fonts inherit from defaults
   )

Analysis Configuration Usage
----------------------------

**FilamentAnalysisConfig**

Configure filament connectivity analysis:

.. code-block:: python

   from lammpskit.config import FilamentAnalysisConfig
   
   # Electrochemical filament analysis
   ecell_config = FilamentAnalysisConfig(
       connectivity_threshold=2.5,  # Ångström cutoff distance
       min_cluster_size=3,          # Minimum atoms per cluster
       gap_analysis=True,           # Include gap size analysis
       temporal_tracking=True,      # Track evolution over time
       statistical_analysis=True   # Include mean/std calculations
   )

**ConnectivityAnalysisConfig**

Network analysis parameters:

.. code-block:: python

   from lammpskit.config import ConnectivityAnalysisConfig
   
   # Network connectivity analysis
   network_config = ConnectivityAnalysisConfig(
       distance_cutoff=3.0,         # Maximum connection distance
       periodic_boundaries=True,    # Account for PBC
       cluster_algorithm='DBSCAN',  # Clustering method
       min_samples=2,               # DBSCAN parameter
       connectivity_metric='euclidean'  # Distance calculation
   )

System Configuration Examples
-----------------------------

**SystemParameters**

Define system-wide analysis parameters:

.. code-block:: python

   from lammpskit.config import SystemParameters
   
   # Molecular dynamics system configuration
   md_system = SystemParameters(
       box_dimensions=[50.0, 50.0, 50.0],  # Simulation box size (Å)
       periodic_boundaries=[True, True, True],  # PBC in x,y,z
       temperature=300.0,    # Target temperature (K)
       pressure=1.0,         # Target pressure (atm)
       timestep=0.001       # Integration timestep (ps)
   )

**MaterialProperties**

Physical material parameters:

.. code-block:: python

   from lammpskit.config import MaterialProperties
   
   # Electrochemical device materials
   device_materials = MaterialProperties(
       electrode_material='Ag',
       electrolyte='TiO2',
       electrode_density=10.49,     # g/cm³
       electrolyte_density=4.23,    # g/cm³
       contact_resistance=100.0,    # Ohm·cm²
       formation_energy=1.2        # eV
   )

Configuration Inheritance
-------------------------

Configuration classes support inheritance for specialized applications:

.. code-block:: python

   # Base configuration for all electrochemical plots
   base_ecell_config = PlotConfig(
       fontsize_title=12,
       fontsize_labels=10,
       format='pdf',
       dpi=300
   )
   
   # Specialized configuration for presentations
   presentation_config = PlotConfig(
       **base_ecell_config.__dict__,  # Inherit base settings
       fontsize_title=16,             # Override title size
       fontsize_labels=14,            # Override label size
       format='svg'                   # Override format
   )

Runtime Configuration Override
------------------------------

Dynamic parameter modification:

.. code-block:: python

   # Start with default configuration
   config = TimeSeriesPlotConfig()
   
   # Override specific parameters for current analysis
   config.alpha = 0.8              # Increase opacity
   config.markersize = 8           # Larger markers
   config.include_line = False     # Remove connecting lines
   
   # Use modified configuration
   create_time_series_plot(x, y, config=config)

Validation and Type Safety
--------------------------

Configuration classes include automatic validation:

.. code-block:: python

   # Type checking prevents common errors
   config = PlotConfig(
       fontsize_title=12,    # ✓ Valid integer
       alpha=0.5,           # ✓ Valid float [0,1]
       format='pdf'         # ✓ Valid output format
   )
   
   # Invalid configurations raise errors at runtime
   invalid_config = PlotConfig(
       alpha=1.5,           # ✗ Invalid: alpha > 1.0
       format='doc'         # ✗ Invalid: unsupported format
   )

Best Practices
--------------

**Configuration Organization**:
   - Create base configurations for project-wide consistency
   - Use inheritance for specialized requirements
   - Document custom configurations with usage examples

**Parameter Selection**:
   - Test font sizes at target output resolution
   - Validate color schemes for colorblind accessibility
   - Consider output medium (screen vs. print) when setting DPI

**Performance Optimization**:
   - Use lower alpha values sparingly (impact rendering speed)
   - Select appropriate marker sizes for data density
   - Choose vector formats for scalable graphics

Related Components
------------------

**Configuration Users**:
   - :mod:`lammpskit.plotting` - All plotting functions accept config objects
   - :mod:`lammpskit.ecellmodel` - Analysis functions use configuration classes
   - :mod:`lammpskit.io` - Output formatting respects configuration settings

**Configuration Extensions**:
   - Custom configuration classes can extend base classes
   - User-specific defaults can override package defaults
   - Plugin modules can define specialized configurations

Module Documentation
--------------------

.. automodule:: lammpskit.config
   :members:
   :show-inheritance:
   :undoc-members:
