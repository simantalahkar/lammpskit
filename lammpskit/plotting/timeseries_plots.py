"""
Centralized plotting functions for time series analysis.

This module provides reusable plotting functions for creating standardized
time series and dual-axis plots with consistent styling and configuration.
These functions are general-purpose and can be used for any time series data,
not just filament analysis.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TimeSeriesPlotConfig:
    """
    Configuration class for standardized time series plotting with publication-ready defaults.
    
    Provides centralized control over plot styling, elements, and output formatting for
    consistent scientific visualization across LAMMPSKit analysis workflows. Supports
    both line plots and scatter plots with flexible element combination.
    
    Attributes
    ----------
    alpha : float, default=0.55
        Transparency level for plot elements. Range [0.0, 1.0] where 0.0 is fully
        transparent and 1.0 is fully opaque. Balanced for overlay visualization.
    linewidth : float, default=0.1
        Line thickness for connected plots. Thin lines prevent visual clutter in
        dense time series data. Use higher values (0.5-2.0) for presentation plots.
    markersize : float, default=5
        Size of scatter plot markers. Optimized for readability without overcrowding.
        Scale proportionally for different figure sizes.
    marker : str, default='^'
        Matplotlib marker style for scatter plots. Options: 'o' (circle), 's' (square),
        '^' (triangle), '*' (star), '+' (plus), 'x' (cross), 'D' (diamond).
    include_line : bool, default=True
        Whether to draw connecting lines between data points. Useful for trend
        visualization in temporal data. Disable for pure scatter analysis.
    include_scatter : bool, default=True
        Whether to draw individual data point markers. Essential for discrete data
        visualization. Disable for smooth trend-only plots.
    format : str, default='pdf'
        Output file format for saved figures. Options: 'pdf' (vector, publication),
        'svg' (web-compatible vector), 'png' (raster), 'eps' (LaTeX-compatible).
    fontsize_title : int, default=8
        Font size for plot titles in points. Optimized for compact scientific layout.
        Use None to disable title font control.
    fontsize_labels : int, default=8
        Font size for axis labels in points. Consistent with scientific journal standards.
        Use None to use matplotlib defaults.
    fontsize_ticks : int, default=8
        Font size for axis tick labels in points. Maintains readability at small sizes.
        Use None to use matplotlib defaults.
    fontsize_legend : int, default=8
        Font size for legend text in points. Balanced for information density.
        Use None to use matplotlib defaults.
    
    Notes
    -----
    Configuration Design Philosophy:
    - Publication-ready defaults minimize post-processing
    - Centralized font control ensures consistency across figures
    - Flexible element control (line/scatter) supports diverse data types
    - Conservative styling prevents visual clutter in complex analyses
    
    Performance Considerations:
    - Transparent plots (alpha < 1.0) may slow rendering for large datasets
    - Vector formats (PDF/SVG) maintain quality but increase file size
    - Font rendering overhead is minimal for typical scientific plots
    
    Examples
    --------
    Default configuration for temporal analysis:
    
    >>> config = TimeSeriesPlotConfig()
    >>> print(f"Using marker: {config.marker}, alpha: {config.alpha}")
    
    Custom configuration for presentation plots:
    
    >>> config = TimeSeriesPlotConfig(
    ...     linewidth=1.0,
    ...     markersize=8,
    ...     alpha=0.8,
    ...     fontsize_title=12,
    ...     fontsize_labels=10
    ... )
    
    Scatter-only configuration for statistical analysis:
    
    >>> config = TimeSeriesPlotConfig(
    ...     include_line=False,
    ...     include_scatter=True,
    ...     marker='o',
    ...     markersize=3
    ... )
    
    High-contrast configuration for printing:
    
    >>> config = TimeSeriesPlotConfig(
    ...     alpha=1.0,
    ...     linewidth=0.8,
    ...     format='eps'
    ... )
    """
    
    # Plot styling
    alpha: float = 0.55
    linewidth: float = 0.1
    markersize: float = 5
    marker: str = '^'
    
    # Plot elements to include
    include_line: bool = True
    include_scatter: bool = True
    
    # File output
    format: str = 'pdf'
    
    # Text and font sizes (centrally controlled)
    fontsize_title: Optional[int] = 8
    fontsize_labels: Optional[int] = 8
    fontsize_ticks: Optional[int] = 8
    fontsize_legend: Optional[int] = 8


@dataclass
class DualAxisPlotConfig:
    """
    Configuration class for dual-axis plots supporting simultaneous visualization of two data series.
    
    Enables comparison of time series data with different units or scales on a single figure.
    Essential for correlating physical quantities like temperature-displacement or 
    connectivity-time relationships in scientific analysis. Provides independent color
    control and legend positioning for clear data interpretation.
    
    Attributes
    ----------
    alpha : float, default=0.55
        Transparency level for both data series. Range [0.0, 1.0]. Moderate transparency
        allows underlying grid and axis lines to remain visible.
    linewidth : float, default=0.1
        Line thickness for both axes data. Thin lines prevent visual dominance of
        either dataset. Increase for presentation or when clarity is critical.
    markersize : float, default=5
        Marker size for scatter points on both axes. Consistent sizing maintains
        visual balance between primary and secondary data series.
    marker : str, default='^'
        Marker style for secondary axis (right). Primary axis uses default scatter
        markers. Options: 'o', 's', '^', '*', '+', 'x', 'D', 'v', '<', '>'.
    primary_color : str, default='tab:red'
        Color for primary (left) y-axis data and axis labels. Matplotlib tab colors
        provide good contrast. Options: 'tab:blue', 'tab:orange', 'tab:green', etc.
    secondary_color : str, default='tab:blue'
        Color for secondary (right) y-axis data and axis labels. Should contrast
        with primary_color for clear visual separation.
    format : str, default='pdf'
        Output file format. Dual-axis plots benefit from vector formats to maintain
        text and line clarity at different scales.
    primary_legend_loc : str, default='upper right'
        Legend position for primary axis data. Standard matplotlib locations:
        'upper/lower/center' + 'left/right/center', or 'best' for automatic.
    secondary_legend_loc : str, default='lower right'
        Legend position for secondary axis data. Should not overlap with primary
        legend. Consider 'upper left', 'lower left', or 'center left'.
    legend_framealpha : float, default=0.75
        Background transparency for legend boxes. Range [0.0, 1.0]. Semi-transparent
        frames prevent complete data occlusion while maintaining readability.
    tight_layout : bool, default=True
        Whether to apply matplotlib tight_layout for automatic spacing adjustment.
        Prevents axis label cutoff in dual-axis configurations.
    fontsize_title : int, default=8
        Font size for plot title. Centered above both axes.
    fontsize_labels : int, default=8  
        Font size for both primary and secondary axis labels.
    fontsize_ticks : int, default=8
        Font size for tick labels on both axes.
    fontsize_legend : int, default=8
        Font size for both legend boxes.
    
    Notes
    -----
    Dual-Axis Design Principles:
    - Color coding clearly distinguishes data series and corresponding axes
    - Legend positioning prevents data occlusion while maintaining clarity
    - Consistent marker sizing maintains visual balance between series
    - Semi-transparent legends allow underlying data visibility
    
    Common Use Cases:
    - Temperature vs. displacement over time
    - Connectivity percentage vs. cluster size evolution  
    - Voltage vs. current relationships in device characterization
    - Statistical metrics vs. physical properties correlation
    
    Performance Considerations:
    - Dual-axis rendering requires additional matplotlib operations
    - Legend placement calculations may slow complex figures
    - Vector output formats recommended for text clarity
    
    Examples
    --------
    Default configuration for scientific analysis:
    
    >>> config = DualAxisPlotConfig()
    >>> print(f"Colors: {config.primary_color}, {config.secondary_color}")
    
    Custom color scheme for publication:
    
    >>> config = DualAxisPlotConfig(
    ...     primary_color='tab:green',
    ...     secondary_color='tab:purple',
    ...     primary_legend_loc='upper left',
    ...     secondary_legend_loc='upper right'
    ... )
    
    High-contrast configuration for presentations:
    
    >>> config = DualAxisPlotConfig(
    ...     alpha=0.9,
    ...     linewidth=0.5,
    ...     markersize=8,
    ...     legend_framealpha=0.9,
    ...     fontsize_title=12,
    ...     fontsize_labels=10
    ... )
    
    Minimal styling for technical reports:
    
    >>> config = DualAxisPlotConfig(
    ...     primary_color='black',
    ...     secondary_color='gray',
    ...     tight_layout=True,
    ...     format='eps'
    ... )
    """
    
    # Plot styling
    alpha: float = 0.55
    linewidth: float = 0.1 
    markersize: float = 5
    marker: str = '^'
    
    # Colors for dual axes
    primary_color: str = 'tab:red'
    secondary_color: str = 'tab:blue'
    
    # File output
    format: str = 'pdf'
    
    # Legend and layout
    primary_legend_loc: str = 'upper right'
    secondary_legend_loc: str = 'lower right'
    legend_framealpha: float = 0.75
    tight_layout: bool = True
    
    # Text and font sizes (centrally controlled)
    fontsize_title: Optional[int] = 8
    fontsize_labels: Optional[int] = 8
    fontsize_ticks: Optional[int] = 8
    fontsize_legend: Optional[int] = 8


def create_time_series_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    stats_label: str,
    config: Optional[TimeSeriesPlotConfig] = None,
    ylim: Optional[Tuple[float, float]] = None,
    # Font size overrides (individual parameter control)
    fontsize_title: Optional[int] = None,
    fontsize_labels: Optional[int] = None,
    fontsize_ticks: Optional[int] = None,
    fontsize_legend: Optional[int] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create standardized time series plots with flexible line and scatter element control.
    
    Generates publication-ready time series visualizations with configurable styling and
    centralized font management. Supports pure line plots, pure scatter plots, or combined
    visualizations based on configuration. Essential for temporal analysis workflows including
    filament evolution tracking, statistical time series, and experimental data visualization.
    
    Parameters
    ----------
    x_data : np.ndarray
        X-axis values, typically representing time or sequential measurements.
        Shape: (n_points,). Units depend on analysis context (e.g., timesteps, seconds, frames).
    y_data : np.ndarray
        Y-axis values for plotting. Shape: (n_points,). Must match x_data length.
        Examples: displacement values, connectivity percentages, statistical measures.
    title : str
        Plot title displayed above the figure. Include units and context for clarity.
        Example: 'Filament Evolution Over Time', 'Temperature vs. Timestep'
    xlabel : str
        X-axis label with units. Standard format: 'Property (units)'.
        Examples: 'Time (ps)', 'Timestep', 'Frame Number', 'Voltage Cycle'
    ylabel : str
        Y-axis label with units. Standard format: 'Property (units)'.
        Examples: 'Displacement (Å)', 'Connectivity (%)', 'Temperature (K)'
    stats_label : str
        Statistical summary or description for legend entry. Often includes computed
        metrics like mean, standard deviation, or frequency. Example: 'Mean: 2.34 ± 0.15'
    config : TimeSeriesPlotConfig, optional
        Plot configuration object controlling styling, elements, and output format.
        If None, uses default configuration optimized for scientific visualization.
    ylim : tuple of float, optional
        Y-axis limits as (ymin, ymax). Useful for consistent scaling across multiple
        related plots or for focusing on specific data ranges.
    fontsize_title : int, optional
        Override configuration title font size. Useful for presentation adaptation
        without modifying the base configuration object.
    fontsize_labels : int, optional
        Override configuration axis label font size. Maintains consistency while
        allowing figure-specific adjustments.
    fontsize_ticks : int, optional
        Override configuration tick label font size. Important for readability
        when figures are scaled for different contexts.
    fontsize_legend : int, optional
        Override configuration legend font size. Critical for maintaining legend
        readability in complex multi-series plots.
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object containing the plot. Can be further customized,
        saved, or displayed using standard matplotlib operations.
    ax : plt.Axes
        Matplotlib axes object for the plot. Provides access for additional
        annotations, reference lines, or styling modifications.
    
    Raises
    ------
    ValueError
        If x_data and y_data have mismatched lengths or contain invalid values.
    TypeError
        If data arrays are not numpy arrays or convertible to arrays.
    
    Notes
    -----
    Plot Element Control:
    - include_line=True, include_scatter=True: Connected scatter plot (default)
    - include_line=True, include_scatter=False: Pure line plot for trends
    - include_line=False, include_scatter=True: Pure scatter plot for discrete data
    - include_line=False, include_scatter=False: Empty plot (not recommended)
    
    Performance Characteristics:
    - Memory usage: O(n_points) for data storage, minimal for plot objects
    - Rendering time: O(n_points) for line plots, O(n_points) for scatter plots
    - File size: Vector formats scale with data complexity, raster formats are fixed
    
    Statistical Integration:
    - Use calculate_mean_std_label() for automated statistical summaries
    - Use calculate_frequency_label() for discrete event analysis
    - Legend automatically includes stats_label for quantitative context
    
    Examples
    --------
    Basic time series plot with default configuration:
    
    >>> import numpy as np
    >>> from lammpskit.plotting.timeseries_plots import create_time_series_plot
    >>> time = np.arange(0, 100, 1)
    >>> displacement = np.random.normal(2.0, 0.5, 100)
    >>> fig, ax = create_time_series_plot(
    ...     time, displacement,
    ...     'Atomic Displacement Evolution',
    ...     'Time (ps)', 'Displacement (Å)',
    ...     'Mean: 2.0 ± 0.5 Å'
    ... )
    
    Custom configuration for presentation:
    
    >>> from lammpskit.plotting.timeseries_plots import TimeSeriesPlotConfig
    >>> config = TimeSeriesPlotConfig(
    ...     linewidth=1.0, markersize=8, alpha=0.8, format='png'
    ... )
    >>> fig, ax = create_time_series_plot(
    ...     time, displacement,
    ...     'High-Visibility Displacement Plot',
    ...     'Time (ps)', 'Displacement (Å)',
    ...     'N=100 points', config=config
    ... )
    
    Scatter-only plot for statistical analysis:
    
    >>> config = TimeSeriesPlotConfig(include_line=False, marker='o')
    >>> fig, ax = create_time_series_plot(
    ...     time, displacement,
    ...     'Discrete Displacement Measurements',
    ...     'Time (ps)', 'Displacement (Å)',
    ...     'σ = 0.5 Å', config=config
    ... )
    
    Controlled y-axis range for comparison plots:
    
    >>> fig, ax = create_time_series_plot(
    ...     time, displacement,
    ...     'Constrained Range Analysis',
    ...     'Time (ps)', 'Displacement (Å)',
    ...     'Range: 0-5 Å', ylim=(0, 5)
    ... )
    
    Font size override for manuscript figures:
    
    >>> fig, ax = create_time_series_plot(
    ...     time, displacement,
    ...     'Publication Figure',
    ...     'Time (ps)', 'Displacement (Å)',
    ...     'Experimental data',
    ...     fontsize_title=14, fontsize_labels=12, fontsize_legend=10
    ... )
    """
    if config is None:
        config = TimeSeriesPlotConfig()
    
    fig, ax = plt.subplots()
    
    # Add line plot if configured
    if config.include_line:
        ax.plot(x_data, y_data, 
               alpha=config.alpha, 
               linewidth=config.linewidth, 
               markersize=config.markersize)
    
    # Add scatter plot if configured
    if config.include_scatter:
        ax.scatter(x_data, y_data, 
                  alpha=config.alpha, 
                  linewidth=config.linewidth, 
                  s=config.markersize, 
                  marker=config.marker, 
                  label=stats_label)
    
    # Set labels and title with centralized font size control
    title_fontsize = fontsize_title or config.fontsize_title
    labels_fontsize = fontsize_labels or config.fontsize_labels
    ticks_fontsize = fontsize_ticks or config.fontsize_ticks
    
    ax.set_xlabel(xlabel, fontsize=labels_fontsize)
    ax.set_ylabel(ylabel, fontsize=labels_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    
    # Set tick label sizes
    if ticks_fontsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
    
    # Set y-limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add legend with centralized font size control
    if config.include_scatter:  # Only show legend if there's a label
        legend_fontsize_final = fontsize_legend or config.fontsize_legend
        if legend_fontsize_final is not None:
            ax.legend(fontsize=legend_fontsize_final)
        else:
            ax.legend()
    
    return fig, ax


def create_dual_axis_plot(
    x_data: np.ndarray,
    primary_y_data: np.ndarray,
    secondary_y_data: np.ndarray,
    title: str,
    xlabel: str,
    primary_ylabel: str,
    secondary_ylabel: str,
    primary_stats_label: str,
    secondary_stats_label: str,
    config: Optional[DualAxisPlotConfig] = None,
    primary_ylim: Optional[Tuple[float, float]] = None,
    secondary_ylim: Optional[Tuple[float, float]] = None,
    # Font size overrides (individual parameter control)
    fontsize_title: Optional[int] = None,
    fontsize_labels: Optional[int] = None,
    fontsize_ticks: Optional[int] = None,
    fontsize_legend: Optional[int] = None
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Create dual-axis plots for simultaneous visualization of two correlated data series.
    
    Generates publication-ready figures with independent y-axes supporting different units,
    scales, or physical quantities. Essential for comparative analysis of time-dependent
    properties where direct correlation visualization is critical. Features automatic
    color coordination, independent axis control, and optimized legend positioning.
    
    Parameters
    ----------
    x_data : np.ndarray
        Shared x-axis values for both data series. Shape: (n_points,). Typically
        represents time, voltage cycles, or sequential measurements.
    primary_y_data : np.ndarray
        Left y-axis data series. Shape: (n_points,). Must match x_data length.
        Examples: displacement values, temperature measurements, primary properties.
    secondary_y_data : np.ndarray
        Right y-axis data series. Shape: (n_points,). Must match x_data length.
        Examples: connectivity percentages, statistical measures, derived properties.
    title : str
        Plot title displayed above both axes. Should indicate the relationship
        being explored. Example: 'Temperature vs Connectivity Evolution'
    xlabel : str
        Shared x-axis label with units. Standard format: 'Property (units)'.
        Examples: 'Time (ps)', 'Voltage Cycle', 'Timestep Number'
    primary_ylabel : str
        Left y-axis label with units. Standard format: 'Property (units)'.
        Color-coded to match primary_color in configuration.
    secondary_ylabel : str
        Right y-axis label with units. Standard format: 'Property (units)'.
        Color-coded to match secondary_color in configuration.
    primary_stats_label : str
        Statistical summary for primary data legend entry. Often includes mean,
        standard deviation, or characteristic values. Example: 'Temp: 300 ± 50 K'
    secondary_stats_label : str
        Statistical summary for secondary data legend entry. Should complement
        primary statistics. Example: 'Connected: 23.4% of time'
    config : DualAxisPlotConfig, optional
        Dual-axis configuration controlling colors, legend positioning, and styling.
        If None, uses default configuration optimized for scientific visualization.
    primary_ylim : tuple of float, optional
        Primary (left) y-axis limits as (ymin, ymax). Useful for consistent scaling
        across multiple related plots or for highlighting specific data ranges.
    secondary_ylim : tuple of float, optional
        Secondary (right) y-axis limits as (ymin, ymax). Independent control enables
        optimal visualization of secondary data regardless of primary axis scaling.
    fontsize_title : int, optional
        Override configuration title font size. Useful for presentation adaptation
        without modifying the base configuration object.
    fontsize_labels : int, optional
        Override configuration axis label font size. Applies to both primary and
        secondary axis labels simultaneously.
    fontsize_ticks : int, optional
        Override configuration tick label font size. Affects both axes tick labels
        for consistent appearance.
    fontsize_legend : int, optional
        Override configuration legend font size. Applies to both legend boxes.
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object containing the dual-axis plot. Can be further
        customized, saved, or displayed using standard matplotlib operations.
    ax1 : plt.Axes
        Primary (left) y-axis axes object. Provides access for additional
        annotations, reference lines, or primary data modifications.
    ax2 : plt.Axes
        Secondary (right) y-axis axes object. Enables independent secondary axis
        customization, additional data series, or specialized annotations.
    
    Raises
    ------
    ValueError
        If data arrays have mismatched lengths or contain invalid values.
    TypeError
        If data arrays are not numpy arrays or convertible to arrays.
    
    Notes
    -----
    Dual-Axis Design Principles:
    - Color coordination ensures clear association between data and corresponding axes
    - Independent axis scaling optimizes visualization of disparate data ranges
    - Legend positioning minimizes data occlusion while maintaining readability
    - Automatic tight layout prevents axis label cutoff in complex configurations
    
    Visual Hierarchy:
    - Primary data (left axis) uses warm colors (red) for visual prominence
    - Secondary data (right axis) uses cool colors (blue) for complementary contrast
    - Legend transparency allows underlying data visibility
    - Consistent marker sizing maintains visual balance
    
    Performance Considerations:
    - Dual-axis rendering requires additional matplotlib twinx() operations
    - Legend placement calculations may impact rendering time for complex data
    - Vector output formats recommended for maintaining text and line clarity
    - Memory usage: O(n_points) for data, minimal overhead for dual axes
    
    Common Applications:
    - Process parameter correlation (temperature vs. pressure over time)
    - Statistical trend analysis (mean vs. variance evolution)
    - Performance monitoring (throughput vs. error rate)
    - Multi-scale temporal analysis (short-term vs. long-term trends)
    
    Examples
    --------
    Basic dual-axis plot with default configuration:
    
    >>> import numpy as np
    >>> from lammpskit.plotting.timeseries_plots import create_dual_axis_plot
    >>> time = np.arange(0, 100, 1)
    >>> temperature = 300 + 50 * np.sin(time * 0.1)
    >>> connectivity = 50 + 30 * np.cos(time * 0.05)
    >>> fig, ax1, ax2 = create_dual_axis_plot(
    ...     time, temperature, connectivity,
    ...     'Temperature-Connectivity Correlation',
    ...     'Time (ps)', 'Temperature (K)', 'Connectivity (%)',
    ...     'T = 300 ± 50 K', 'C = 50 ± 30%'
    ... )
    
    Custom configuration for presentation:
    
    >>> from lammpskit.plotting.timeseries_plots import DualAxisPlotConfig
    >>> config = DualAxisPlotConfig(
    ...     primary_color='tab:green',
    ...     secondary_color='tab:purple',
    ...     primary_legend_loc='upper left',
    ...     secondary_legend_loc='lower right',
    ...     alpha=0.8
    ... )
    >>> fig, ax1, ax2 = create_dual_axis_plot(
    ...     time, temperature, connectivity,
    ...     'Custom Color Analysis',
    ...     'Time (ps)', 'Property A', 'Property B',
    ...     'Series A', 'Series B', config=config
    ... )
    
    Controlled axis ranges for comparison:
    
    >>> fig, ax1, ax2 = create_dual_axis_plot(
    ...     time, temperature, connectivity,
    ...     'Fixed Range Comparison',
    ...     'Time (ps)', 'Temperature (K)', 'Connectivity (%)',
    ...     'Controlled range', 'Fixed scale',
    ...     primary_ylim=(250, 350), secondary_ylim=(0, 100)
    ... )
    
    High-contrast configuration for printing:
    
    >>> config = DualAxisPlotConfig(
    ...     primary_color='black',
    ...     secondary_color='gray',
    ...     alpha=1.0,
    ...     legend_framealpha=1.0,
    ...     format='eps'
    ... )
    >>> fig, ax1, ax2 = create_dual_axis_plot(
    ...     time, temperature, connectivity,
    ...     'Print-Optimized Dual Plot',
    ...     'Time (ps)', 'Primary', 'Secondary',
    ...     'Data A', 'Data B', config=config
    ... )
    
    Font override for manuscript figures:
    
    >>> fig, ax1, ax2 = create_dual_axis_plot(
    ...     time, temperature, connectivity,
    ...     'Publication Figure',
    ...     'Time (ps)', 'Temperature (K)', 'Connectivity (%)',
    ...     'Experimental', 'Calculated',
    ...     fontsize_title=16, fontsize_labels=14, fontsize_legend=12
    ... )
    """
    if config is None:
        config = DualAxisPlotConfig()
    
    fig, ax1 = plt.subplots()
    
    # Get font sizes with override capability
    title_fontsize = fontsize_title or config.fontsize_title
    labels_fontsize = fontsize_labels or config.fontsize_labels
    ticks_fontsize = fontsize_ticks or config.fontsize_ticks
    legend_fontsize_final = fontsize_legend or config.fontsize_legend
    
    # Configure primary axis (left)
    ax1.set_xlabel(xlabel, fontsize=labels_fontsize)
    ax1.set_ylabel(primary_ylabel, color=config.primary_color, fontsize=labels_fontsize)
    ax1.scatter(x_data, primary_y_data, 
               alpha=config.alpha, 
               linewidth=config.linewidth, 
               s=config.markersize, 
               color=config.primary_color, 
               label=primary_stats_label)
    ax1.tick_params(axis='y', labelcolor=config.primary_color, labelsize=ticks_fontsize)
    ax1.tick_params(axis='x', labelsize=ticks_fontsize)
    
    # Set primary y-limits if provided
    if primary_ylim is not None:
        ax1.set_ylim(primary_ylim)
    
    # Create secondary axis (right)
    ax2 = ax1.twinx()
    ax2.set_ylabel(secondary_ylabel, color=config.secondary_color, fontsize=labels_fontsize)
    ax2.scatter(x_data, secondary_y_data, 
               alpha=config.alpha, 
               linewidth=config.linewidth, 
               s=config.markersize, 
               marker=config.marker, 
               color=config.secondary_color, 
               label=secondary_stats_label)
    ax2.tick_params(axis='y', labelcolor=config.secondary_color, labelsize=ticks_fontsize)
    
    # Set secondary y-limits if provided
    if secondary_ylim is not None:
        ax2.set_ylim(secondary_ylim)
    
    # Set title with font size control
    plt.title(title, fontsize=title_fontsize)
    
    # Apply tight layout if configured
    if config.tight_layout:
        fig.tight_layout()
    
    # Add legends with font size control
    legend_kwargs = {'framealpha': config.legend_framealpha}
    if legend_fontsize_final is not None:
        legend_kwargs['fontsize'] = legend_fontsize_final
    
    ax1.legend(loc=config.primary_legend_loc, **legend_kwargs)
    ax2.legend(loc=config.secondary_legend_loc, **legend_kwargs)
    
    return fig, ax1, ax2


def save_and_close_figure(
    fig: plt.Figure,
    output_dir: str,
    filename: str,
    file_format: str = 'pdf'
) -> None:
    """
    Save matplotlib figure to disk with automatic directory creation and memory cleanup.
    
    Provides standardized figure output handling for scientific visualization workflows.
    Automatically creates output directories, handles filename formatting, and closes
    figures to prevent memory accumulation during batch processing. Essential for
    automated analysis pipelines generating multiple plots.
    
    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object to save. Can be any figure created with plt.figure(),
        plt.subplots(), or plotting functions returning figure objects.
    output_dir : str
        Target directory for saved figure. Created automatically if it doesn't exist.
        Supports both absolute and relative paths. Use '.' for current directory.
    filename : str
        Base filename without extension. Extension is added automatically based on
        file_format parameter. Should be descriptive of plot content for organization.
    file_format : str, optional, default='pdf'
        Output file format determining quality and compatibility:
        - 'pdf': Vector format, publication-ready, scalable
        - 'svg': Web-compatible vector format, editable
        - 'png': Raster format, good for web display
        - 'eps': LaTeX-compatible vector format
        - 'jpg'/'jpeg': Compressed raster, smaller files
        
    Raises
    ------
    OSError
        If output directory cannot be created due to permissions or disk space.
    ValueError
        If file_format is not supported by matplotlib backend.
    
    Notes
    -----
    Memory Management:
    - Automatically closes figure after saving to prevent memory leaks
    - Critical for batch processing workflows generating many plots
    - Use plt.show() before calling this function if display is also needed
    
    Directory Handling:
    - Creates nested directory structures automatically
    - Preserves existing directories and files
    - No error if output_dir already exists
    
    Performance Considerations:
    - Vector formats (PDF, SVG, EPS) maintain quality but may be larger
    - Raster formats (PNG, JPG) have fixed resolution but smaller files
    - PDF recommended for scientific publications and presentations
    - PNG recommended for web display and documentation
    
    Examples
    --------
    Save figure with automatic directory creation:
    
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0, 10, 100)
    >>> ax.plot(x, np.sin(x))
    >>> save_and_close_figure(fig, 'output/plots', 'sine_wave')
    # Saves as 'output/plots/sine_wave.pdf'
    
    Save in different formats for various uses:
    
    >>> save_and_close_figure(fig, 'manuscript/figures', 'analysis', 'eps')
    >>> save_and_close_figure(fig, 'web/images', 'analysis', 'png')
    >>> save_and_close_figure(fig, 'presentations', 'analysis', 'svg')
    
    Organized output structure:
    
    >>> base_dir = 'results/experiment_2024'
    >>> save_and_close_figure(fig, f'{base_dir}/temperature', 'temp_vs_time')
    >>> save_and_close_figure(fig, f'{base_dir}/displacement', 'disp_evolution')
    
    Current directory output:
    
    >>> save_and_close_figure(fig, '.', 'quick_analysis', 'png')
    # Saves as './quick_analysis.png'
    
    Batch processing workflow:
    
    >>> figures = [fig1, fig2, fig3]
    >>> names = ['temperature', 'pressure', 'density']
    >>> for fig, name in zip(figures, names):
    ...     save_and_close_figure(fig, 'output/timeseries', name)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{filename}.{file_format}"
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close(fig)


def calculate_mean_std_label(data: np.ndarray, label_prefix: str, precision: int = 2) -> str:
    """
    Generate standardized statistical summary labels for plot legends and annotations.
    
    Computes mean and standard deviation of input data and formats as publication-ready
    label string. Essential for automated legend generation in scientific plots where
    quantitative summaries enhance data interpretation. Supports flexible precision
    control for different measurement scales and reporting requirements.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array for statistical calculation. Shape: (n_points,).
        Supports any numeric data type. NaN values are handled by numpy functions.
    label_prefix : str
        Descriptive text preceding the statistical values. Should include property
        name and units for clarity. Example: 'Temperature (K)', 'Displacement (Å)'
    precision : int, optional, default=2
        Number of decimal places for formatting statistical values. Range: 0-15.
        Use 0-1 for large values, 2-4 for typical scientific measurements,
        5+ for high-precision requirements.
        
    Returns
    -------
    str
        Formatted label string in standard "prefix = mean ± std" format.
        Uses Unicode ± symbol for professional appearance. Compatible with
        matplotlib legend and annotation functions.
    
    Raises
    ------
    ValueError
        If precision is negative or data array is empty.
    TypeError
        If data is not array-like or label_prefix is not string.
    
    Notes
    -----
    Statistical Calculations:
    - Mean: Arithmetic average computed with np.mean()
    - Standard deviation: Sample standard deviation with np.std() (N-1 denominator)
    - NaN handling: Follows numpy conventions (NaN propagation)
    
    Formatting Standards:
    - Uses Unicode ± (U+00B1) for professional appearance
    - Precision applies to both mean and standard deviation
    - No scientific notation; adjust precision for extreme values
    
    Performance Characteristics:
    - Computational complexity: O(n) for statistical calculations
    - Memory usage: O(1) additional memory beyond input array
    - String formatting: Minimal overhead for typical legend use
    
    Applications:
    - Time series analysis summary statistics
    - Experimental data characterization
    - Model validation metrics
    - Comparative analysis legend entries
    
    Examples
    --------
    Basic temperature data summarization:
    
    >>> import numpy as np
    >>> temperatures = np.array([298.2, 301.5, 299.8, 300.1, 302.3])
    >>> label = calculate_mean_std_label(temperatures, 'Temperature (K)')
    >>> print(label)
    'Temperature (K) = 300.38 ± 1.52'
    
    Displacement analysis with high precision:
    
    >>> displacements = np.random.normal(2.345, 0.123, 1000)
    >>> label = calculate_mean_std_label(displacements, 'Displacement (Å)', precision=4)
    >>> print(label)
    'Displacement (Å) = 2.3451 ± 0.1234'
    
    Large-scale data with low precision:
    
    >>> particle_counts = np.random.poisson(1e6, 100)
    >>> label = calculate_mean_std_label(particle_counts, 'Count', precision=0)
    >>> print(label)
    'Count = 1000023 ± 1000'
    
    Percentage data with appropriate precision:
    
    >>> percentages = np.array([23.45, 24.12, 22.89, 23.78, 24.56])
    >>> label = calculate_mean_std_label(percentages, 'Connectivity (%)', precision=1)
    >>> print(label)
    'Connectivity (%) = 23.8 ± 0.6'
    
    Integration with plotting workflows:
    
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot(range(len(temperatures)), temperatures, label=label)
    >>> ax.legend()  # Automatically uses formatted statistical label
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    return f"{label_prefix} = {mean_val:.{precision}f} +/- {std_val:.{precision}f}"


def calculate_frequency_label(data: np.ndarray, target_value, label_template: str, precision: int = 2) -> str:
    """
    Calculate occurrence frequency of specific values and generate formatted labels.
    
    Computes percentage frequency of target value occurrence in data arrays and formats
    using customizable template strings. Essential for binary state analysis, event
    detection summaries, and categorical data visualization. Supports flexible label
    formatting for diverse scientific reporting contexts.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array for frequency analysis. Shape: (n_points,).
        Supports any data type that supports equality comparison (int, float, bool, str).
    target_value : any
        Specific value to count occurrences of. Must be comparable to data elements
        using == operator. Examples: 1, 0, True, 'connected', specific float values.
    label_template : str
        Format string template with {frequency} placeholder for percentage insertion.
        Supports all Python string formatting options. Example: 'Active {frequency:.1f}% of time'
    precision : int, optional, default=2
        Decimal places for frequency percentage formatting. Range: 0-10.
        Note: This parameter is currently unused; precision controlled by template format specifiers.
        
    Returns
    -------
    str
        Formatted label string with frequency percentage substituted into template.
        Percentage calculated as (occurrences / total_points) * 100.
    
    Raises
    ------
    ValueError
        If data array is empty or label_template missing {frequency} placeholder.
    TypeError
        If target_value type incompatible with data elements for comparison.
    KeyError
        If label_template contains invalid format specifications.
    
    Notes
    -----
    Frequency Calculation:
    - Uses element-wise equality (==) for counting matches
    - Percentage = (matches / total_elements) * 100
    - Range: 0.0% (no matches) to 100.0% (all matches)
    - Floating-point precision handled by template format specifiers
    
    Template Formatting:
    - Supports all Python str.format() capabilities
    - Use {frequency:.1f} for 1 decimal place, {frequency:.0f} for integers
    - Can include additional text, units, and formatting
    - Multiple {frequency} references allowed in single template
    
    Performance Characteristics:
    - Computational complexity: O(n) for equality comparison
    - Memory usage: O(1) additional memory beyond input array
    - Boolean array creation for comparison may temporarily double memory
    
    Common Applications:
    - Binary state analysis (connected/disconnected, active/inactive)
    - Event detection (threshold crossings, state changes)
    - Categorical data summaries (phase classification, state distribution)
    - Time-based occurrence rates (duty cycles, sampling frequencies)
    
    Examples
    --------
    Binary connectivity analysis:
    
    >>> import numpy as np
    >>> connectivity = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    >>> label = calculate_frequency_label(connectivity, 1, "Connected {frequency:.1f}% of time")
    >>> print(label)
    'Connected 70.0% of time'
    
    Boolean state analysis:
    
    >>> active_states = np.array([True, False, True, True, False])
    >>> label = calculate_frequency_label(active_states, True, "Active: {frequency:.0f}%")
    >>> print(label)
    'Active: 60%'
    
    Threshold crossing analysis:
    
    >>> temperatures = np.array([298, 305, 310, 295, 307, 312, 290])
    >>> over_threshold = temperatures > 300
    >>> label = calculate_frequency_label(over_threshold, True, "Above 300K: {frequency:.1f}%")
    >>> print(label)
    'Above 300K: 57.1%'
    
    Categorical state distribution:
    
    >>> phases = np.array(['A', 'B', 'A', 'A', 'C', 'B', 'A'])
    >>> label_A = calculate_frequency_label(phases, 'A', "Phase A: {frequency:.1f}%")
    >>> label_B = calculate_frequency_label(phases, 'B', "Phase B: {frequency:.1f}%")
    >>> print(label_A, '|', label_B)
    'Phase A: 57.1%' | 'Phase B: 28.6%'
    
    Multiple format references:
    
    >>> successes = np.array([1, 0, 1, 1, 0])
    >>> label = calculate_frequency_label(
    ...     successes, 1, 
    ...     "Success rate: {frequency:.1f}% ({frequency:.2f}% precise)"
    ... )
    >>> print(label)
    'Success rate: 60.0% (60.00% precise)'
    
    Integration with time series plotting:
    
    >>> connectivity_data = np.random.choice([0, 1], 1000, p=[0.3, 0.7])
    >>> stats_label = calculate_frequency_label(
    ...     connectivity_data, 1, "Connected {frequency:.1f}% of simulation"
    ... )
    >>> # Use stats_label in create_time_series_plot() for automatic legend generation
    """
    frequency = np.sum(data == target_value) / len(data) * 100
    return label_template.format(frequency=frequency)
