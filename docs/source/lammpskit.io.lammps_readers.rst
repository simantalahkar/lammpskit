lamLAMMPS trajectory file readers for efficient data loading and processing. This module provides optimized functions for reading LAMMPS dump files with support for large trajectories and selective data loading.

Key Functions
-------------

The module provides functions for:

- **Structure information reading** from LAMMPS dump files
- **Coordinate data reading** with selective column loading
- **Memory-efficient processing** for large trajectory files
- **Standard LAMMPS format support** with flexible parsinglammps\_readers module
===================================

LAMMPS trajectory file readers for molecular dynamics analysis. This module provides the core I/O functionality for parsing LAMMPS dump files and extracting simulation metadata with robust error handling.

Functions Overview
------------------

Core functions for LAMMPS trajectory file reading:

- **read_structure_info** - Extract metadata from LAMMPS dump files
- **read_coordinates** - Load atomic coordinates with selective data access

File Format Support
-------------------

Supports standard LAMMPS dump format with headers:

.. code-block:: text

   ITEM: TIMESTEP
   ITEM: NUMBER OF ATOMS  
   ITEM: BOX BOUNDS [units]
   ITEM: ATOMS [column headers]

**Standard Column Layout**: ``id type charge x y z vx vy vz fx fy fz``

Usage Examples
--------------

Extract simulation metadata:

.. code-block:: python

   from lammpskit.io import read_structure_info
   
   timestep, atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_structure_info('dump.lammpstrj')
   electrode_separation = zhi - zlo

Load coordinates for time-series analysis:

.. code-block:: python

   import glob
   from lammpskit.io import read_coordinates
   from lammpskit.config import DEFAULT_COLUMNS_TO_READ
   
   files = sorted(glob.glob('dump.*.lammpstrj'))
   coords, timesteps, atoms, *box = read_coordinates(
       files, skip_rows=9, columns_to_read=DEFAULT_COLUMNS_TO_READ)

Error Handling
--------------

Functions use fail-fast approach with descriptive error messages:

- :exc:`FileNotFoundError` - File doesn't exist or access denied
- :exc:`EOFError` - Truncated or malformed trajectory file
- :exc:`ValueError` - Invalid column indices or non-numeric data

Related Functions
-----------------

- :func:`lammpskit.config.validate_file_list` - Validate trajectory file lists
- :func:`lammpskit.ecellmodel.filament_layer_analysis.read_displacement_data` - Read processed displacement data

Module Documentation
--------------------

.. automodule:: lammpskit.io.lammps_readers
   :members:
   :show-inheritance:
   :undoc-members:
