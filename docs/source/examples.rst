Examples
========

This section provides practical examples demonstrating LAMMPSKit capabilities for various analysis workflows.

Basic Examples
--------------

Reading Trajectory Files
~~~~~~~~~~~~~~~~~~~~~~~~

Extract simulation metadata from a single trajectory file:

.. code-block:: python

   from lammpskit.io import read_structure_info
   
   # Read basic simulation parameters
   timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_structure_info('dump.lammpstrj')
   
   print(f"Timestep: {timestep}")
   print(f"Number of atoms: {total_atoms}")
   print(f"Box dimensions: X=[{xlo:.1f}, {xhi:.1f}], Y=[{ylo:.1f}, {yhi:.1f}], Z=[{zlo:.1f}, {zhi:.1f}]")
   
   # Calculate electrode separation for electrochemical cells
   electrode_separation = zhi - zlo
   print(f"Electrode separation: {electrode_separation:.1f} Å")

Loading Coordinate Data
~~~~~~~~~~~~~~~~~~~~~~~

Load atomic coordinates from multiple trajectory files:

.. code-block:: python

   import glob
   from lammpskit.io import read_coordinates
   from lammpskit.config import DEFAULT_COLUMNS_TO_READ
   
   # Get sorted list of trajectory files
   files = sorted(glob.glob('dump.*.lammpstrj'))
   
   # Load coordinates with standard column configuration
   coords, timesteps, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(
       files, skip_rows=9, columns_to_read=DEFAULT_COLUMNS_TO_READ)
   
   print(f"Loaded {len(files)} files with {coords.shape}")
   print(f"Timestep range: {timesteps[0]} to {timesteps[-1]}")
   print(f"Coordinate array shape: {coords.shape}")  # (n_files, n_atoms, n_columns)

General Plotting
~~~~~~~~~~~~~~~~

Create comparative plots for multiple datasets:

.. code-block:: python

   import numpy as np
   from lammpskit.plotting import plot_multiple_cases
   
   # Generate sample data - atomic distributions along z-axis
   z_positions = np.linspace(-10, 40, 50)
   hf_distribution_set = np.random.poisson(10, 50)      # Hafnium atoms in SET state
   hf_distribution_reset = np.random.poisson(5, 50)     # Hafnium atoms in RESET state
   
   # Combine data for multi-case plotting
   distributions = np.array([hf_distribution_set, hf_distribution_reset])
   labels = ['SET state', 'RESET state']
   
   # Create comparative plot
   fig = plot_multiple_cases(
       distributions, z_positions, labels,
       'Hf atom count', 'Z position (Å)', 
       'hafnium_distribution', 10, 8,
       output_dir='./plots'
   )

Electrochemical Analysis Examples
---------------------------------

Atomic Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze spatial distributions of different atom types:

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_distribution
   
   # Define analysis parameters
   file_list = ['set_state.lammpstrj', 'reset_state.lammpstrj']
   labels = ['SET', 'RESET']
   z_bins = [-15, 45, 60]  # [z_min, z_max, n_bins]
   
   # Perform atomic distribution analysis
   plot_atomic_distribution(
       file_list=file_list,
       labels=labels,
       skip_rows=9,
       z_bins=z_bins,
       analysis_name='voltage_comparison',
       output_dir='./analysis_output',
       columns_to_read=DEFAULT_COLUMNS_TO_READ
   )
   
   # This generates plots for:
   # - Hafnium (Hf) distribution
   # - Oxygen (O) distribution  
   # - Tantalum (Ta) distribution
   # - Metal atom distribution
   # - Stoichiometry analysis

Charge Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze atomic charge distributions across the electrochemical cell:

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_charge_distribution
   
   # Charge distribution analysis with custom parameters
   plot_atomic_charge_distribution(
       file_list=['trajectory_1.lammpstrj', 'trajectory_2.lammpstrj'],
       labels=['0.5V', '1.0V'],
       skip_rows=9,
       z_bins=[-10, 50, 80],  # Higher resolution binning
       analysis_name='charge_evolution',
       output_dir='./charge_analysis',
       columns_to_read=EXTENDED_COLUMNS_TO_READ  # Include charge data
   )

Filament Evolution Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track filament connectivity and structural evolution over time:

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import track_filament_evolution
   
   # Time series analysis parameters
   trajectory_files = sorted(glob.glob('evolution_*.lammpstrj'))
   TIME_STEP = 0.001  # ps per MD step
   DUMP_INTERVAL_STEPS = 500  # steps between trajectory dumps
   
   # Track filament properties over time
   track_filament_evolution(
       file_list=trajectory_files,
       analysis_name='filament_evolution',
       time_step=TIME_STEP,
       dump_interval_steps=DUMP_INTERVAL_STEPS,
       output_dir='./evolution_analysis',
       columns_to_read=DEFAULT_COLUMNS_TO_READ
   )
   
   # Generates plots for:
   # - Connectivity state over time
   # - Filament gap distance evolution
   # - Filament separation changes
   # - Cluster size distribution

Cluster Analysis
~~~~~~~~~~~~~~~~

Perform OVITO-based cluster analysis for filament characterization:

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters
   
   # Single-file cluster analysis
   cluster_results = analyze_clusters('filament_snapshot.lammpstrj')
   
   # The function automatically:
   # - Identifies metallic atom clusters
   # - Calculates connectivity across electrodes
   # - Determines filament gap distances
   # - Analyzes cluster size distributions
   # - Saves visualization images

Displacement Analysis
~~~~~~~~~~~~~~~~~~~~~

Compare atomic displacements between different simulation conditions:

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_comparison
   
   # Displacement data files (processed LAMMPS output)
   displacement_files = [
       'displacement_low_temp.dat',
       'displacement_high_temp.dat',
       'displacement_medium_temp.dat'
   ]
   
   labels = ['300K', '600K', '450K']
   
   # Compare displacement profiles
   plot_displacement_comparison(
       file_list=displacement_files,
       loop_start=0,
       loop_end=1000,
       labels=labels,
       analysis_name='temperature_comparison',
       repeat_count=5,  # Number of repeated measurements
       output_dir='./displacement_analysis'
   )

Advanced Examples
-----------------

Time Series Plotting
~~~~~~~~~~~~~~~~~~~~

Create customized time series plots with dual axes:

.. code-block:: python

   import numpy as np
   from lammpskit.plotting import create_dual_axis_plot, DualAxisPlotConfig
   
   # Generate sample time series data
   time = np.linspace(0, 100, 200)  # Time in ps
   connectivity = np.random.rand(200) * 100  # Connectivity percentage
   temperature = 300 + 50 * np.sin(2 * np.pi * time / 20)  # Temperature oscillation
   
   # Configure dual-axis plot
   config = DualAxisPlotConfig(
       primary_color='tab:red',
       secondary_color='tab:blue',
       alpha=0.7,
       linewidth=0.5
   )
   
   # Create dual-axis plot
   fig, ax1, ax2 = create_dual_axis_plot(
       x_data=time,
       primary_y_data=connectivity,
       secondary_y_data=temperature,
       title='Filament Evolution vs Temperature',
       xlabel='Time (ps)',
       primary_ylabel='Connectivity (%)',
       secondary_ylabel='Temperature (K)',
       primary_stats_label='Connectivity: Mean=45.2%, Std=12.1%',
       secondary_stats_label='Temperature: Mean=315.4K, Std=25.8K',
       config=config
   )

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handle large datasets with memory optimization:

.. code-block:: python

   from lammpskit.config import DEFAULT_COLUMNS_TO_READ
   from lammpskit.io import read_coordinates
   
   # For very large trajectory sets, process in batches
   all_files = sorted(glob.glob('large_trajectory_*.lammpstrj'))
   batch_size = 10
   
   for i in range(0, len(all_files), batch_size):
       batch_files = all_files[i:i+batch_size]
       
       # Use core columns only to reduce memory usage
       core_columns = (0, 1, 2, 3, 4, 5)  # id, type, charge, x, y, z
       
       coords, timesteps, atoms, *box = read_coordinates(
           batch_files, skip_rows=9, columns_to_read=core_columns)
       
       # Process this batch
       print(f"Processed batch {i//batch_size + 1}: {coords.shape}")
       
       # Perform analysis on this batch...
       # analyze_batch(coords, timesteps, atoms, box)

Complete Workflow Example
-------------------------

The ``usage/ecellmodel/run_analysis.py`` script demonstrates a comprehensive analysis workflow:

.. code-block:: python

   #!/usr/bin/env python3
   """
   Complete LAMMPSKit analysis workflow example.
   
   This script demonstrates:
   1. Filament evolution tracking
   2. Displacement analysis  
   3. Charge distribution analysis
   4. Atomic distribution analysis
   """
   
   import os
   import sys
   import glob
   from pathlib import Path
   
   # Add package to path if running from repository
   if __name__ == "__main__":
       repo_root = Path(__file__).parent.parent.parent
       sys.path.insert(0, str(repo_root))
   
   from lammpskit.ecellmodel.filament_layer_analysis import (
       track_filament_evolution,
       plot_atomic_distribution, 
       plot_atomic_charge_distribution,
       plot_displacement_comparison
   )
   from lammpskit.config import EXTENDED_COLUMNS_TO_READ
   
   def main():
       """Main analysis workflow."""
       output_dir = os.path.join(".", "usage", "ecellmodel", "output")
       
       # Analysis Block 1: Filament Evolution Tracking
       TIME_STEP = 0.001
       DUMP_INTERVAL_STEPS = 500
       
       data_path = os.path.join(".", "usage", "ecellmodel", "data", "trajectory_series", "*.lammpstrj")
       file_list = sorted(glob.glob(data_path))
       
       if file_list:
           track_filament_evolution(
               file_list=file_list,
               analysis_name='evolution_tracking',
               time_step=TIME_STEP,
               dump_interval_steps=DUMP_INTERVAL_STEPS,
               output_dir=output_dir,
               columns_to_read=EXTENDED_COLUMNS_TO_READ
           )
       
       # Additional analysis blocks for displacement, charge, and atomic distributions...
   
   if __name__ == "__main__":
       main()

This example demonstrates how to structure a complete analysis pipeline using LAMMPSKit's modular architecture.
