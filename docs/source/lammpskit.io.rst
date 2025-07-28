lammpskit.io package
====================

The ``lammpskit.io`` package provides essential I/O functionality for reading and parsing LAMMPS molecular dynamics simulation output files. This module is designed for robust handling of trajectory data with comprehensive error checking and memory-efficient processing.

Key Features
------------

- **Robust LAMMPS dump file parsing** with automatic format detection
- **Memory-efficient coordinate loading** with selective column reading
- **Batch processing capabilities** for time-series analysis
- **Comprehensive error handling** with descriptive failure messages
- **Support for large datasets** with optimized memory management

Core Functions
--------------

The I/O module provides two primary functions for trajectory data access:

- **read_structure_info** - Extract metadata from LAMMPS dump files
- **read_coordinates** - Load atomic coordinates with selective data access

Performance Considerations
--------------------------

Memory usage scales as O(F × N × C) where:
- F = number of files
- N = number of atoms  
- C = number of columns read

For large datasets (>1GB), use :data:`~lammpskit.config.DEFAULT_COLUMNS_TO_READ` instead of :data:`~lammpskit.config.EXTENDED_COLUMNS_TO_READ` to reduce memory footprint by ~60%.

Submodules
----------

.. toctree::
   :maxdepth: 4

   lammpskit.io.lammps_readers

Module contents
---------------

.. automodule:: lammpskit.io
   :members:
   :show-inheritance:
   :undoc-members:
