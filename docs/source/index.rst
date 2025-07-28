LAMMPSKit Documentation
=======================

**LAMMPSKit** is a Python toolkit for running, post-processing, and analyzing molecular dynamics (MD) simulations with LAMMPS. It provides a collection of functions and scripts to streamline simulation workflows and automate analysis of LAMMPS output data.

Features
--------

- **Modular architecture** with separate I/O, plotting, and analysis components
- **Comprehensive data processing** for LAMMPS trajectory and displacement files
- **Advanced visualization tools** including timeseries plotting with font customization
- **Atomic-scale analysis functions** for distributions, charges, displacements, and clusters
- **Filament evolution tracking** for electrochemical simulation analysis
- **Configuration management** with centralized settings and validation
- **Complete example workflows** demonstrating real-world usage patterns
- **Extensive test coverage** with visual regression testing for plots

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install lammpskit

For development:

.. code-block:: bash

   pip install .[dev]

Quick Start
-----------

Basic workflow for electrochemical analysis:

.. code-block:: python

   import lammpskit as lk
   from lammpskit.config import DEFAULT_COLUMNS_TO_READ
   
   # Load trajectory data
   file_list = ['trajectory1.lammpstrj', 'trajectory2.lammpstrj']
   coords, timesteps, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = lk.io.read_coordinates(
       file_list, skip_rows=9, columns_to_read=DEFAULT_COLUMNS_TO_READ)
   
   # Perform cluster analysis
   from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters
   analyze_clusters('trajectory1.lammpstrj')

Package Structure
-----------------

LAMMPSKit features a modular architecture:

- **lammpskit.io** - Data reading and I/O operations
- **lammpskit.plotting** - Visualization utilities and timeseries plots  
- **lammpskit.ecellmodel** - Electrochemical analysis functions
- **lammpskit.config** - Centralized configuration management

Target Applications
-------------------

- Electrochemical memory device simulations (ReRAM, memristors)
- Ion transport and defect migration studies
- Phase transition analysis in oxide materials
- General LAMMPS trajectory post-processing

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api_reference
   examples
   data_formats

API Reference
=============

.. toctree::
   :maxdepth: 3
   :caption: API Documentation:
   
   lammpskit
   lammpskit.config
   lammpskit.io
   lammpskit.plotting
   lammpskit.ecellmodel

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
