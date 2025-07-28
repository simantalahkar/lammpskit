Data Formats
============

This section describes the data formats used by LAMMPSKit for input and output files.

LAMMPS Trajectory Files
-----------------------

LAMMPS Dump Format (.lammpstrj)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LAMMPSKit reads standard LAMMPS dump files with the following structure:

.. code-block:: text

   ITEM: TIMESTEP
   1200000
   ITEM: NUMBER OF ATOMS
   5000
   ITEM: BOX BOUNDS pp pp pp
   0.0 50.0
   0.0 50.0
   0.0 50.0
   ITEM: ATOMS id type q x y z ix iy iz vx vy vz c_eng
   1 2 0.1 1.0 2.0 3.0 0 0 0 0.1 0.2 0.3 -1.5
   2 1 -0.2 2.0 3.0 4.0 0 0 0 0.0 0.1 0.2 -1.2
   3 4 0.0 3.0 4.0 5.0 0 0 0 -0.1 0.0 0.1 -1.8
   ... (one line per atom)

Header Structure
^^^^^^^^^^^^^^^^

1. **TIMESTEP**: Simulation timestep number
2. **NUMBER OF ATOMS**: Total atom count in the system
3. **BOX BOUNDS**: Simulation cell dimensions (xlo xhi, ylo yhi, zlo zhi)
4. **ATOMS**: Column headers for atomic data

Standard Column Layout
^^^^^^^^^^^^^^^^^^^^^^

The typical LAMMPS dump format includes these columns:

.. list-table:: Standard LAMMPS Columns
   :header-rows: 1
   :widths: 10 20 15 55

   * - Index
     - Column
     - Units
     - Description
   * - 0
     - id
     - --
     - Unique atom identifier
   * - 1
     - type
     - --
     - Atom type (species identifier)
   * - 2
     - q
     - e
     - Atomic charge
   * - 3
     - x
     - Å
     - X-coordinate
   * - 4
     - y
     - Å
     - Y-coordinate
   * - 5
     - z
     - Å
     - Z-coordinate
   * - 6
     - ix
     - --
     - Image flag (x-direction)
   * - 7
     - iy
     - --
     - Image flag (y-direction)
   * - 8
     - iz
     - --
     - Image flag (z-direction)
   * - 9
     - vx
     - Å/ps
     - Velocity (x-component)
   * - 10
     - vy
     - Å/ps
     - Velocity (y-component)
   * - 11
     - vz
     - Å/ps
     - Velocity (z-component)
   * - 12
     - fx
     - eV/Å
     - Force (x-component)
   * - 13
     - fy
     - eV/Å
     - Force (y-component)
   * - 14
     - fz
     - eV/Å
     - Force (z-component)
   * - 15
     - c_eng
     - eV
     - Potential energy per atom

Column Configuration Constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LAMMPSKit provides predefined column selections:

.. code-block:: python

   from lammpskit.config import DEFAULT_COLUMNS_TO_READ, EXTENDED_COLUMNS_TO_READ
   
   # Core analysis columns: id, type, charge, x, y, z, vx, vy, vz, fx
   DEFAULT_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)
   
   # Extended columns for comprehensive analysis  
   EXTENDED_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16)

Atom Type System
~~~~~~~~~~~~~~~~

For HfTaO electrochemical cell simulations, LAMMPSKit uses this atom type mapping:

.. list-table:: Atom Type Mapping
   :header-rows: 1
   :widths: 20 30 50

   * - Type ID
     - Element
     - Description
   * - 2
     - Hafnium (Hf)
     - Primary filament-forming species
   * - 1, 3, 5, 7, 9, ...
     - Oxygen (O)
     - Odd-numbered types (oxygen anions)
   * - 4, 6, 8, 10, ...
     - Tantalum (Ta)
     - Even-numbered types except 2
   * - 5, 6, 9, 10
     - Electrodes
     - Also function as electrode atoms

Usage in Analysis Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Hafnium atoms (filament-forming species)
   hf_mask = (atom_types == 2)
   
   # Oxygen atoms (anion species)
   oxygen_mask = (atom_types % 2 == 1)
   
   # Tantalum atoms (excluding hafnium)
   tantalum_mask = ((atom_types % 2 == 0) & (atom_types != 2))
   
   # Electrode atoms
   electrode_mask = np.isin(atom_types, [5, 6, 9, 10])

Displacement Data Files
-----------------------

Processed Displacement Format (.dat)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LAMMPSKit can read processed displacement data files with this structure:

.. code-block:: text

   # Header line 1: analysis parameters
   # Header line 2: data description  
   # Header line 3: column information
   0 2
   1.0 3.5 0.8 2.1 1.2 0.9
   2.0 4.2 0.9 2.3 1.4 1.1
   3.0 5.1 1.1 2.5 1.6 1.3
   # end loop
   1 3
   1.1 3.7 0.7 2.2 1.3 1.0
   2.1 4.4 0.8 2.4 1.5 1.2
   # end loop

Data Structure
^^^^^^^^^^^^^^

- **Header lines**: Begin with '#' and contain metadata
- **Loop markers**: Integer pairs marking data sections (start_index end_index)
- **Data lines**: Space-separated numeric values
- **End markers**: '# end loop' terminates each data section

Displacement Data Labels
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lammpskit.config import DISPLACEMENT_DATA_LABELS
   
   # Available displacement data types:
   DISPLACEMENT_DATA_LABELS = [
       'abs total disp',        # Total displacement magnitude (Å)
       'density - mass',        # Mass density (g/cm³)
       'temp (K)',             # Temperature (Kelvin)  
       'z disp (A)',           # Vertical displacement (Å)
       'lateral disp (A)',     # Horizontal displacement (Å)
       'outward disp vector (A)' # Radial displacement (Å)
   ]

Reading Displacement Data
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lammpskit.ecellmodel.filament_layer_analysis import read_displacement_data
   
   # Read displacement data with loop parameters
   bin_position, displacement_data = read_displacement_data(
       filepath='displacement_analysis.dat',
       loop_start=0,
       loop_end=1000,
       repeat_count=5
   )

Output Formats
--------------

Plot Output Formats
~~~~~~~~~~~~~~~~~~~

LAMMPSKit generates plots in multiple formats:

**PDF Format (default)**:
- Vector format suitable for publications
- Scalable without quality loss
- Small file size for simple plots

**SVG Format**:
- Web-compatible vector format
- Editable in graphic design software
- Good for interactive applications

**PNG Format** (optional):
- Raster format for presentations
- Fixed resolution
- Universal compatibility

Configuration Example
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from lammpskit.plotting import TimeSeriesPlotConfig
   
   # Configure output format
   config = TimeSeriesPlotConfig(format='pdf')  # or 'svg', 'png', 'eps'

Analysis Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

LAMMPSKit creates organized output directories:

.. code-block:: text

   output_dir/
   ├── atomic_distribution_hf.pdf
   ├── atomic_distribution_o.pdf  
   ├── atomic_distribution_ta.pdf
   ├── charge_distribution_hf.pdf
   ├── filament_connectivity.pdf
   ├── filament_gap_evolution.pdf
   └── displacement_comparison.pdf

File Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^

- ``{analysis_name}_{data_type}.{format}``
- ``{element_type}_distribution.{format}``
- ``filament_{property}_evolution.{format}``
- ``displacement_{comparison_type}.{format}``

Memory and Performance Considerations
-------------------------------------

File Size Guidelines
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Typical File Sizes
   :header-rows: 1
   :widths: 30 20 25 25

   * - File Type
     - Atoms
     - Files
     - Memory Usage
   * - Single trajectory
     - 5,000
     - 1
     - ~1 MB
   * - Time series (10 columns)
     - 5,000
     - 100
     - ~50 MB
   * - Large system
     - 50,000
     - 100
     - ~500 MB
   * - Extended analysis
     - 50,000
     - 1,000
     - ~5 GB

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

1. **Column Selection**: Use ``DEFAULT_COLUMNS_TO_READ`` vs ``EXTENDED_COLUMNS_TO_READ``
2. **Batch Processing**: Process files in smaller groups for large datasets
3. **Memory Management**: Clear coordinate arrays between analysis steps
4. **File Format**: Use compressed formats for storage (gzip) when possible

.. code-block:: python

   # Memory-efficient approach for large datasets
   for batch in file_batches:
       coords, timesteps, atoms, *box = read_coordinates(
           batch, skip_rows=9, columns_to_read=DEFAULT_COLUMNS_TO_READ)
       # Process batch
       del coords  # Free memory between batches
