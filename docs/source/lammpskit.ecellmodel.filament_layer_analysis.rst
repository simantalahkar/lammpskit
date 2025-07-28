lammpskit.ecellmodel.filament\_layer\_analysis module
=====================================================

Specialized analysis functions for filament formation and connectivity evolution in electrochemical devices. This module provides comprehensive tools for analyzing conductive filament formation, percolation pathways, and device switching behavior in molecular dynamics simulations.

Key Functions
-------------

The module provides specialized functions for:

- **Displacement data reading** from LAMMPS trajectory files
- **Atomic distribution plotting** with statistical analysis
- **Charge distribution plotting** for electrical characterization
- **Displacement comparison analysis** across multiple datasets
- **Cluster analysis** for connectivity determination
- **Filament evolution tracking** over time
- **Time series plotting** for temporal analysis

Filament Analysis Workflow
--------------------------

Complete workflow for filament formation analysis:

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import (
       analyze_filament_connectivity,
       track_filament_evolution,
       classify_device_state,
       plot_connectivity_evolution
   )
   
   # 1. Analyze connectivity across all timesteps
   connectivity_data = analyze_filament_connectivity(
       trajectory_data,
       distance_threshold=2.5,        # Å, connectivity cutoff
       electrode_regions={
           'bottom': (0, 5),          # Bottom electrode z-range
           'top': (45, 50)            # Top electrode z-range
       },
       min_cluster_size=3,            # Minimum atoms per cluster
       periodic_boundaries=True       # Account for PBC
   )
   
   # 2. Track temporal evolution
   evolution_metrics = track_filament_evolution(
       connectivity_data,
       analysis_window=100,           # Timesteps for local analysis
       trend_detection=True,          # Detect formation/dissolution trends
       switching_detection=True       # Identify switching events
   )
   
   # 3. Classify device states
   device_states = classify_device_state(
       connectivity_data,
       hrs_threshold=0.1,             # Low connectivity = HRS
       lrs_threshold=0.8,             # High connectivity = LRS
       intermediate_range=(0.1, 0.8) # Intermediate resistance states
   )
   
   # 4. Visualize results
   plot_connectivity_evolution(
       connectivity_data,
       device_states=device_states,
       save_path='filament_analysis.pdf',
       include_statistics=True
   )

Connectivity Analysis Examples
-----------------------------

**Basic Connectivity Analysis**:

.. code-block:: python

   # Analyze atomic connectivity with distance-based criteria
   connectivity = analyze_filament_connectivity(
       trajectory_data,
       distance_threshold=2.5,        # Connectivity distance (Å)
       analysis_type='full',          # Full connectivity analysis
       include_clusters=True,         # Include cluster analysis
       include_pathways=True          # Include percolation pathways
   )
   
   # Results include:
   # - connectivity_matrix: atom-atom connectivity
   # - cluster_labels: cluster assignments for each atom
   # - pathway_analysis: percolation pathway identification
   # - connectivity_ratio: fraction of connected atoms

**Advanced Connectivity with Electrode Regions**:

.. code-block:: python

   # Define electrode regions for device analysis
   electrode_config = {
       'bottom': {
           'z_range': (0, 5),         # Bottom electrode region
           'material': 'Ag',          # Electrode material
           'bias': -1.0              # Applied bias (V)
       },
       'top': {
           'z_range': (45, 50),       # Top electrode region
           'material': 'Ag',          # Electrode material
           'bias': 1.0               # Applied bias (V)
       }
   }
   
   # Analyze electrode-to-electrode connectivity
   device_connectivity = find_electrode_connections(
       trajectory_data,
       electrode_config,
       pathway_algorithm='shortest_path',  # Pathfinding algorithm
       resistance_model='ohmic',           # Resistance calculation model
       include_intermediate=True           # Include intermediate states
   )

**Cluster Analysis with Multiple Algorithms**:

.. code-block:: python

   # Compare different clustering approaches
   cluster_comparison = analyze_cluster_distribution(
       connectivity_data,
       algorithms=['dbscan', 'connected_components', 'hierarchical'],
       dbscan_params={'eps': 2.5, 'min_samples': 3},
       hierarchical_params={'distance_threshold': 3.0},
       comparison_metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz']
   )

Temporal Evolution Analysis
---------------------------

**Filament Formation Tracking**:

.. code-block:: python

   # Track filament growth and dissolution dynamics
   evolution = track_filament_evolution(
       connectivity_time_series,
       detection_params={
           'formation_threshold': 0.1,    # Connectivity increase threshold
           'dissolution_threshold': 0.9,  # Connectivity decrease threshold
           'minimum_duration': 50,        # Minimum event duration (timesteps)
           'noise_filter': True           # Apply noise filtering
       },
       analysis_metrics=[
           'growth_rate',                 # Filament formation speed
           'dissolution_rate',            # Filament dissolution speed
           'stability_index',             # State stability measure
           'switching_frequency'          # Number of switches per unit time
       ]
   )

**Gap Evolution Analysis**:

.. code-block:: python

   # Analyze gap formation and healing in partially connected devices
   gap_evolution = track_gap_evolution(
       trajectory_data,
       gap_definition={
           'max_distance': 5.0,           # Maximum gap distance (Å)
           'min_gap_size': 1.0,           # Minimum detectable gap (Å)
           'electrode_separation': 50.0   # Device thickness (Å)
       },
       tracking_params={
           'temporal_resolution': 10,     # Analysis every 10 timesteps
           'spatial_resolution': 0.5,     # Spatial binning (Å)
           'persistence_threshold': 20    # Minimum gap lifetime (timesteps)
       }
   )

Device State Classification
---------------------------

**Multi-State Classification**:

.. code-block:: python

   # Classify device into multiple resistance states
   state_classification = classify_device_state(
       connectivity_data,
       classification_scheme={
           'HRS': (0.0, 0.1),            # High resistance state
           'IRS1': (0.1, 0.4),           # Intermediate resistance state 1
           'IRS2': (0.4, 0.7),           # Intermediate resistance state 2
           'LRS': (0.7, 1.0)             # Low resistance state
       },
       state_stability={
           'minimum_duration': 100,       # Minimum state duration (timesteps)
           'transition_hysteresis': 0.05, # Hysteresis for state transitions
           'noise_tolerance': 0.02        # Noise tolerance for classification
       }
   )

**Switching Event Detection**:

.. code-block:: python

   # Detect and characterize switching events
   switching_analysis = calculate_switching_metrics(
       device_states,
       switching_criteria={
           'minimum_state_change': 2,     # Minimum state difference for switching
           'transition_window': 50,       # Window for transition detection (timesteps)
           'confirmation_window': 20      # Confirmation of stable new state
       },
       performance_metrics=[
           'switching_speed',             # Time for state transition
           'switching_voltage',           # Voltage at switching
           'energy_consumption',          # Energy per switching event
           'reliability_index'            # Switching success rate
       ]
   )

Gap Analysis and Critical Path Finding
--------------------------------------

**Comprehensive Gap Analysis**:

.. code-block:: python

   # Analyze gap distribution and critical bottlenecks
   gap_analysis = analyze_gap_distribution(
       trajectory_data,
       gap_parameters={
           'search_radius': 5.0,          # Search radius for gaps (Å)
           'gap_threshold': 2.0,          # Minimum gap size (Å)
           'electrode_distance': 50.0     # Total electrode separation (Å)
       },
       analysis_options={
           'include_histograms': True,    # Generate gap size histograms
           'include_spatial_maps': True,  # Create spatial gap maps
           'include_critical_paths': True, # Find shortest connection paths
           'temporal_tracking': True     # Track gap evolution over time
       }
   )

**Critical Path Analysis**:

.. code-block:: python

   # Find critical gaps that prevent electrode connection
   critical_gaps = find_critical_gaps(
       connectivity_data,
       pathfinding_params={
           'algorithm': 'dijkstra',       # Pathfinding algorithm
           'weight_function': 'distance', # Edge weight calculation
           'max_search_depth': 100,       # Maximum search iterations
           'path_optimization': True      # Optimize found paths
       },
       gap_criteria={
           'minimum_impact': 0.1,         # Minimum connectivity impact
           'spatial_clustering': True,    # Group nearby gaps
           'temporal_persistence': 50     # Minimum gap lifetime (timesteps)
       }
   )

Visualization and Reporting
---------------------------

**Connectivity Evolution Plots**:

.. code-block:: python

   # Create comprehensive connectivity evolution plots
   plot_connectivity_evolution(
       connectivity_data,
       plot_options={
           'include_states': True,        # Show device state regions
           'include_statistics': True,    # Add statistical summaries
           'include_events': True,        # Mark switching events
           'color_scheme': 'professional' # Publication-ready colors
       },
       time_series_config={
           'temporal_smoothing': 10,      # Smoothing window (timesteps)
           'confidence_intervals': True,  # Show confidence bands
           'trend_analysis': True         # Include trend lines
       }
   )

**Filament Structure Visualization**:

.. code-block:: python

   # Visualize filament structure and connectivity
   plot_filament_structure(
       connectivity_data,
       visualization_params={
           'view_angle': 'side',          # Viewing perspective
           'highlight_pathways': True,    # Highlight percolation pathways
           'color_by_cluster': True,      # Color atoms by cluster
           'show_electrodes': True        # Show electrode regions
       },
       export_options={
           'format': 'pdf',              # Output format
           'resolution': 300,            # DPI for raster elements
           'vector_graphics': True       # Use vector graphics when possible
       }
   )

Performance Optimization
------------------------

**Memory-Efficient Analysis**:

.. code-block:: python

   # Analyze large trajectories with memory management
   for timestep_batch in process_trajectory_batches(
       trajectory_file,
       batch_size=1000,               # Process 1000 timesteps at once
       overlap=100,                   # Overlap between batches
       memory_limit='8GB'             # Maximum memory usage
   ):
       batch_connectivity = analyze_filament_connectivity(
           timestep_batch,
           distance_threshold=2.5,
           parallel_processing=True,
           n_cores=4
       )

**Parallel Processing**:

.. code-block:: python

   # Parallelize analysis across multiple CPU cores
   parallel_results = analyze_connectivity_parallel(
       trajectory_data,
       n_cores=8,                     # Number of CPU cores
       chunk_size=500,                # Timesteps per chunk
       load_balancing=True,           # Enable dynamic load balancing
       shared_memory=True             # Use shared memory for efficiency
   )

Research Applications
---------------------

**Mechanism Studies**:
   - Filament formation kinetics analysis
   - Ion migration pathway identification
   - Electric field effect characterization
   - Temperature dependence investigation

**Device Optimization**:
   - Electrode material selection
   - Device geometry optimization
   - Operating condition determination
   - Reliability enhancement strategies

**Material Design**:
   - Electrolyte property correlation
   - Interface engineering analysis
   - Defect impact assessment
   - Dopant concentration optimization

Integration Examples
--------------------

**With Data Processing Module**:

.. code-block:: python

   from lammpskit.ecellmodel.data_processing import load_trajectory_data
   
   # Load and preprocess trajectory data
   trajectory = load_trajectory_data('simulation.lammpstrj')
   processed_data = preprocess_for_filament_analysis(trajectory)
   
   # Perform filament analysis
   connectivity = analyze_filament_connectivity(processed_data)

**With Plotting System**:

.. code-block:: python

   from lammpskit.plotting import create_dual_axis_plot
   
   # Create dual-axis plot for connectivity vs. temperature
   fig, ax1, ax2 = create_dual_axis_plot(
       x_data=time,
       primary_y_data=connectivity_ratio,
       secondary_y_data=temperature,
       title='Filament Connectivity vs Temperature',
       primary_ylabel='Connectivity Ratio',
       secondary_ylabel='Temperature (K)'
   )

Related Functions
-----------------

- :mod:`lammpskit.ecellmodel.data_processing` - Core data processing utilities
- :mod:`lammpskit.plotting.timeseries_plots` - Time series visualization
- :mod:`lammpskit.config` - Configuration management for analysis parameters
- :mod:`lammpskit.io` - Trajectory file loading and processing

Module Documentation
--------------------

.. automodule:: lammpskit.ecellmodel.filament_layer_analysis
   :members:
   :show-inheritance:
   :undoc-members:
