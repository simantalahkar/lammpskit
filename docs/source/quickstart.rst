Quick Start
===========

This guide demonstrates the basic usage of LAMMPSKit for analyzing LAMMPS molecular dynamics simulation data.

Basic Workflow
--------------

1. **Import the package**

.. code-block:: python

   import lammpskit as lk
   from lammpskit.config import DEFAULT_COLUMNS_TO_READ

2. **Read trajectory data**

.. code-block:: python

   # Single file metadata
   timestep, atoms, xlo, xhi, ylo, yhi, zlo, zhi = lk.io.read_structure_info('dump.lammpstrj')
   
   # Multiple files for time series
   file_list = ['dump.100000.lammpstrj', 'dump.200000.lammpstrj', 'dump.300000.lammpstrj']
   coords, timesteps, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = lk.io.read_coordinates(
       file_list, skip_rows=9, columns_to_read=DEFAULT_COLUMNS_TO_READ)

3. **Perform analysis**

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import (
       analyze_clusters, plot_atomic_distribution
   )
   
   # Cluster analysis for filament connectivity
   analyze_clusters('dump.lammpstrj')
   
   # Atomic distribution analysis
   z_bins = [-10, 40, 50]  # z-direction binning
   plot_atomic_distribution(file_list, ['Case 1'], skip_rows=9, 
                           z_bins=z_bins, analysis_name='atomic_dist', 
                           output_dir='./output')

4. **Create plots**

.. code-block:: python

   import numpy as np
   
   # General plotting utility
   x_data = np.array([1, 2, 3, 4, 5])
   y_data = np.array([[1, 4, 9, 16, 25], [1, 8, 27, 64, 125]])
   labels = ['Linear', 'Cubic']
   
   fig = lk.plotting.plot_multiple_cases(
       x_data, y_data, labels, 'X values', 'Y values', 
       'comparison', 8, 6, output_dir='./plots')

Example Workflow
----------------

LAMMPSKit includes a comprehensive example workflow in ``usage/ecellmodel/``:

.. code-block:: bash

   # Clone repository and navigate to examples
   git clone https://github.com/simantalahkar/lammpskit.git
   cd lammpskit/usage/ecellmodel
   
   # Run complete analysis workflow
   python run_analysis.py

This demonstrates four main analysis types:

1. **Filament Evolution Tracking** - Monitor connectivity over time
2. **Displacement Analysis** - Compare atomic displacements  
3. **Charge Distribution Analysis** - Analyze local charge distributions
4. **Atomic Distribution Analysis** - Study distributions under different voltages

Key Concepts
------------

Atom Type System
~~~~~~~~~~~~~~~~

LAMMPSKit uses a specific atom type mapping for HfTaO electrochemical cells:

- **Type 2**: Hafnium (Hf) atoms
- **Odd types (1, 3, 5, 7, 9, ...)**: Oxygen (O) atoms
- **Even types except 2 (4, 6, 8, 10, ...)**: Tantalum (Ta) atoms  
- **Types 5, 6, 9, 10**: Also function as electrode atoms

Column Configuration
~~~~~~~~~~~~~~~~~~~~

LAMMPS dump files typically contain columns: ``id type charge x y z vx vy vz fx fy fz``

.. code-block:: python

   from lammpskit.config import DEFAULT_COLUMNS_TO_READ, EXTENDED_COLUMNS_TO_READ
   
   # Core analysis columns: id, type, charge, x, y, z, fx, fy, fz, extras
   DEFAULT_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)
   
   # Extended columns for comprehensive analysis
   EXTENDED_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16)

File Formats
~~~~~~~~~~~~

**LAMMPS Trajectory (.lammpstrj)**:

.. code-block:: text

   ITEM: TIMESTEP
   1200000
   ITEM: NUMBER OF ATOMS
   5000
   ITEM: BOX BOUNDS pp pp pp
   0.0 50.0
   0.0 50.0
   0.0 50.0
   ITEM: ATOMS id type q x y z vx vy vz fx fy fz
   1 2 0.1 1.0 2.0 3.0 0 0 0 0.1 0.2 0.3
   ...

Memory Considerations
---------------------

For large datasets (>1GB):

- Use ``DEFAULT_COLUMNS_TO_READ`` instead of ``EXTENDED_COLUMNS_TO_READ``
- Process files in smaller batches
- Consider chunked reading for very large trajectories

Performance scales as O(F × N × C) where F=files, N=atoms, C=columns.
