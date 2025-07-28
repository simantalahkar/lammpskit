{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

Examples
--------

Basic usage:

.. code-block:: python

   from {{ module }} import {{ objname }}
   
   # Example usage for {{ objname }}
   result = {{ objname }}(input_data)

{% if objname in ['analyze_clusters', 'track_filament_evolution'] %}

Scientific context for filament analysis:

.. code-block:: python

   # Electrochemical device analysis workflow
   connectivity_data = {{ objname }}(
       trajectory_data,
       distance_threshold=2.5,    # Å, based on ionic radii
       min_cluster_size=3,        # Minimum percolating unit
       electrode_regions=['bottom', 'top']
   )
   
   # Results provide device state classification
   if connectivity_data['percolation_ratio'] > 0.8:
       device_state = 'LRS'  # Low resistance state
   else:
       device_state = 'HRS'  # High resistance state

{% elif objname.startswith('plot_') %}

Visualization example:

.. code-block:: python

   # Create publication-ready plots
   fig = {{ objname }}(
       data,
       title='{{ objname | replace("_", " ") | title }}',
       save_path='analysis_{{ objname }}.pdf',
       config=plot_config
   )

{% endif %}

Notes
-----

This function is part of the LAMMPSKit electrochemical analysis toolkit. 
For comprehensive analysis workflows, see :mod:`lammpskit.ecellmodel.filament_layer_analysis`.

See Also
--------
{% if objname == 'analyze_clusters' %}
:func:`track_filament_evolution` : Track temporal changes in clusters
:func:`plot_atomic_distribution` : Visualize cluster spatial distribution
{% elif objname == 'track_filament_evolution' %}
:func:`analyze_clusters` : Analyze cluster connectivity
:func:`plot_displacement_timeseries` : Visualize temporal evolution
{% elif objname.startswith('plot_') %}
:mod:`lammpskit.plotting` : General plotting utilities
:mod:`lammpskit.config` : Configuration classes for plots
{% endif %}
