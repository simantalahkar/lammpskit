{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members: __init__

{% if objname == 'TimeSeriesPlotConfig' %}

Attributes
----------

alpha : float
    Transparency level for plot elements (0.0-1.0)
linewidth : float  
    Width of plotted lines
markersize : int
    Size of plot markers
marker : str
    Marker style for scatter plots
include_line : bool
    Whether to include connecting lines
include_scatter : bool
    Whether to include scatter points
format : str
    Output format ('pdf', 'svg', 'png')
fontsize_title : int
    Font size for plot titles
fontsize_labels : int
    Font size for axis labels
fontsize_ticks : int
    Font size for tick labels
fontsize_legend : int
    Font size for legend text

Examples
--------

Basic time series configuration:

.. code-block:: python

   from lammpskit.plotting import TimeSeriesPlotConfig, create_time_series_plot
   
   # Standard scientific publication style
   config = TimeSeriesPlotConfig(
       alpha=0.7,
       linewidth=1.0,
       markersize=6,
       marker='o',
       format='pdf'
   )
   
   # Use in time series plotting
   fig, ax = create_time_series_plot(
       x_data=time,
       y_data=connectivity,
       title='Filament Connectivity Evolution',
       xlabel='Time (ps)',
       ylabel='Connectivity (%)',
       config=config
   )

Presentation-ready configuration:

.. code-block:: python

   # Larger fonts and markers for presentations
   presentation_config = TimeSeriesPlotConfig(
       fontsize_title=16,
       fontsize_labels=14,
       fontsize_ticks=12,
       markersize=8,
       linewidth=2.0,
       format='svg'
   )

{% elif objname == 'DualAxisPlotConfig' %}

Attributes
----------

primary_color : str
    Color for primary y-axis data
secondary_color : str
    Color for secondary y-axis data
primary_legend_loc : str
    Legend location for primary axis
secondary_legend_loc : str  
    Legend location for secondary axis
legend_framealpha : float
    Transparency of legend background
tight_layout : bool
    Whether to use tight layout

Examples
--------

Dual-axis correlation analysis:

.. code-block:: python

   from lammpskit.plotting import DualAxisPlotConfig, create_dual_axis_plot
   
   # Configure colors and legend positions
   config = DualAxisPlotConfig(
       primary_color='tab:red',
       secondary_color='tab:blue', 
       primary_legend_loc='upper left',
       secondary_legend_loc='upper right'
   )
   
   # Create correlation plot
   fig, ax1, ax2 = create_dual_axis_plot(
       x_data=time,
       primary_y_data=connectivity,
       secondary_y_data=temperature,
       title='Connectivity vs Temperature',
       xlabel='Time (ps)',
       primary_ylabel='Connectivity (%)',
       secondary_ylabel='Temperature (K)',
       config=config
   )

{% endif %}

Notes
-----

Configuration classes use Python dataclasses for type safety and default value management.
All font sizes are in points (pt) and follow scientific publication standards.

See Also
--------
{% if objname == 'TimeSeriesPlotConfig' %}
:func:`create_time_series_plot` : Function that uses this configuration
:class:`DualAxisPlotConfig` : Configuration for dual-axis plots
{% elif objname == 'DualAxisPlotConfig' %}
:func:`create_dual_axis_plot` : Function that uses this configuration  
:class:`TimeSeriesPlotConfig` : Configuration for single-axis plots
{% endif %}
:mod:`lammpskit.config` : General configuration constants
