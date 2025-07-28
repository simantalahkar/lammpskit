lammpskit.ecellmodel.data\_processing module
============================================

Core data processing utilities for electrochemical device analysis. This module provides fundamental data handling, statistical analysis, and preprocessing functions for molecular dynamics simulation data in electrochemical device contexts.

Key Functions
-------------

The module provides functions for:

- **Atom type selection** from coordinate data
- **Z-direction binning setup** for layered analysis
- **Atomic distribution calculations** for spatial analysis
- **Charge distribution calculations** for electrical analysis
- **Element label extraction** from filenames

Data Processing Workflow
------------------------

Typical data processing pipeline for electrochemical device analysis:

.. code-block:: python

   from lammpskit.ecellmodel.data_processing import (
       load_trajectory_data,
       validate_data_integrity,
       calculate_connectivity_statistics,
       compute_temporal_correlations
   )
   
   # 1. Load raw trajectory data
   trajectory = load_trajectory_data(
       'simulation.lammpstrj',
       columns=['id', 'type', 'x', 'y', 'z'],
       timestep_range=(0, 10000)
   )
   
   # 2. Validate data quality
   validation_report = validate_data_integrity(
       trajectory,
       check_continuity=True,
       check_boundaries=True,
       report_missing=True
   )
   
   # 3. Calculate statistical properties
   connectivity_stats = calculate_connectivity_statistics(
       trajectory,
       distance_threshold=2.5,
       electrode_regions=['bottom', 'top']
   )
   
   # 4. Analyze temporal correlations
   correlations = compute_temporal_correlations(
       connectivity_stats,
       lag_range=(1, 100),
       correlation_method='pearson'
   )

Statistical Analysis Examples
-----------------------------

**Connectivity Statistics**:

.. code-block:: python

   # Calculate comprehensive connectivity metrics
   stats = calculate_connectivity_statistics(
       trajectory_data,
       distance_threshold=2.5,        # Å, connectivity cutoff
       min_cluster_size=3,            # Minimum atoms per cluster
       electrode_separation=50.0,     # Å, device thickness
       periodic_boundaries=True       # Account for PBC
   )
   
   # Results include:
   # - connectivity_ratio: fraction of connected atoms
   # - cluster_size_distribution: histogram of cluster sizes
   # - percolation_probability: likelihood of electrode connection
   # - gap_size_statistics: analysis of non-connected regions

**Device Performance Metrics**:

.. code-block:: python

   # Calculate switching and performance characteristics
   performance = calculate_switching_metrics(
       connectivity_time_series,
       hrs_threshold=0.1,             # High resistance state cutoff
       lrs_threshold=0.8,             # Low resistance state cutoff
       switching_time_window=1000,    # Timesteps for switching detection
       noise_filter=True              # Apply noise reduction
   )
   
   # Performance metrics:
   # - switching_ratio: HRS/LRS resistance ratio
   # - switching_speed: transition time (timesteps)
   # - retention_time: state stability duration
   # - endurance_cycles: number of successful switches

**Temporal Correlation Analysis**:

.. code-block:: python

   # Analyze temporal relationships in device behavior
   correlations = compute_temporal_correlations(
       device_metrics,
       properties=['connectivity', 'temperature', 'potential'],
       lag_range=(1, 200),            # Correlation time range
       significance_level=0.05        # Statistical significance
   )
   
   # Correlation results:
   # - autocorrelation_functions: property self-correlation
   # - cross_correlation_matrix: inter-property correlations
   # - characteristic_timescales: decay time constants
   # - significant_lags: statistically significant correlations

Data Quality and Validation
---------------------------

**Data Integrity Checking**:

.. code-block:: python

   # Comprehensive data validation
   validation = validate_data_integrity(
       trajectory_data,
       checks={
           'continuity': True,        # Check for missing timesteps
           'boundaries': True,        # Validate coordinate ranges
           'atom_conservation': True, # Verify atom count consistency
           'energy_conservation': True, # Check energy drift
           'temperature_stability': True # Validate thermostat performance
       },
       tolerance_levels={
           'position': 0.01,          # Å, maximum position drift
           'energy': 0.1,             # eV, maximum energy drift
           'temperature': 5.0         # K, maximum temperature variation
       }
   )

**Missing Data Handling**:

.. code-block:: python

   # Interpolate missing timesteps
   complete_data = interpolate_missing_timesteps(
       trajectory_data,
       method='linear',               # Interpolation method
       max_gap=10,                   # Maximum interpolatable gap
       extrapolate=False             # Don't extrapolate beyond data
   )
   
   # Filter noisy data
   clean_data = filter_noise_data(
       trajectory_data,
       filter_type='gaussian',       # Noise filter type
       sigma=1.0,                    # Filter parameter
       preserve_features=True        # Maintain important features
   )

Performance Optimization
------------------------

**Memory Management**:

.. code-block:: python

   # Stream large trajectory files
   for timestep_data in load_trajectory_data(
       'large_simulation.lammpstrj',
       stream=True,                  # Enable streaming
       chunk_size=1000,              # Timesteps per chunk
       memory_limit='4GB'            # Maximum memory usage
   ):
       process_timestep_data(timestep_data)

**Computational Efficiency**:

.. code-block:: python

   # Optimize processing for large datasets
   processed_data = aggregate_temporal_data(
       trajectory_data,
       aggregation_window=100,       # Aggregate every 100 timesteps
       parallel_processing=True,     # Use multiprocessing
       n_cores=4,                    # Number of CPU cores
       cache_results=True           # Cache intermediate results
   )

Integration Examples
--------------------

**With Filament Analysis**:

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import analyze_filament_connectivity
   
   # Prepare data for filament analysis
   processed_trajectory = normalize_trajectory_data(
       raw_trajectory,
       center_coordinates=True,
       scale_time=True
   )
   
   # Perform filament analysis
   filament_data = analyze_filament_connectivity(
       processed_trajectory,
       connectivity_threshold=2.5
   )

**With Plotting System**:

.. code-block:: python

   from lammpskit.plotting import create_time_series_plot
   
   # Extract time series for plotting
   time_series = extract_time_series(
       processed_data,
       property='connectivity_ratio',
       time_units='ps'
   )
   
   # Create standardized plot
   fig, ax = create_time_series_plot(
       x_data=time_series['time'],
       y_data=time_series['connectivity'],
       title='Device Connectivity Evolution',
       xlabel='Time (ps)',
       ylabel='Connectivity Ratio'
   )

Common Use Cases
----------------

**Device Characterization**:
   - Resistance state identification
   - Switching threshold determination
   - Performance parameter extraction
   - Device stability analysis

**Research Applications**:
   - Filament formation mechanism analysis
   - Material property correlation studies
   - Temperature dependence investigations
   - Applied field effect characterization

**Quality Control**:
   - Simulation convergence verification
   - Data consistency validation
   - Error detection and reporting
   - Reproducibility assessment

Error Handling and Diagnostics
------------------------------

**Exception Handling**:

.. code-block:: python

   try:
       trajectory = load_trajectory_data('simulation.lammpstrj')
   except FileNotFoundError:
       logger.error("Trajectory file not found")
   except MemoryError:
       logger.warning("File too large, enabling streaming mode")
       trajectory = load_trajectory_data('simulation.lammpstrj', stream=True)
   except DataValidationError as e:
       logger.error(f"Data validation failed: {e.message}")

**Diagnostic Information**:

.. code-block:: python

   # Generate processing diagnostics
   diagnostics = generate_processing_diagnostics(
       trajectory_data,
       include_memory_usage=True,
       include_timing=True,
       include_quality_metrics=True
   )

Related Functions
-----------------

- :mod:`lammpskit.ecellmodel.filament_layer_analysis` - Specialized filament analysis
- :mod:`lammpskit.io` - File I/O operations
- :mod:`lammpskit.plotting` - Visualization functions
- :mod:`lammpskit.config` - Configuration management

Module Documentation
--------------------

.. automodule:: lammpskit.ecellmodel.data_processing
   :members:
   :show-inheritance:
   :undoc-members:
