"""
LAMMPSKit Configuration Module
=============================

Centralized configuration settings for plotting, data processing, and analysis
across the LAMMPSKit package.

This module provides constants and configuration classes to ensure consistency
across different analysis functions and reduce code duplication.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


# =============================================================================
# PLOTTING CONSTANTS
# =============================================================================

# Color schemes and styling
DEFAULT_COLORS = ['b', 'r', 'g', 'k']
DEFAULT_LINESTYLES = ['--', '-.', ':', '-']
DEFAULT_MARKERS = ['o', 's', '^', 'v']

# Font sizes for different plot elements
FONT_SIZES = {
    'legend': 8,
    'label_small': 5,
    'label_medium': 7,
    'label_large': 8,
    'title': 8,
    'tick_major': 6,
    'tick_minor': 5,
}

# Plot styling defaults
PLOT_DEFAULTS = {
    'markersize': 5,
    'linewidth': 1.2,
    'alpha': 0.75,
    'legend_location': 'upper center',
    'bbox_inches': 'tight',
}

# Figure layout defaults
LAYOUT_DEFAULTS = {
    'constrained_layout': False,
    'squeeze': False,
    'tight_layout': True,
    'subplot_adjust': {'hspace': 0},
}

# File output formats
OUTPUT_FORMATS = ['pdf', 'svg']
DEFAULT_OUTPUT_FORMAT = 'svg'


# =============================================================================
# DATA PROCESSING CONSTANTS  
# =============================================================================

# Standard data reading configurations
COLUMNS_TO_READ_DEFAULT = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)
COLUMNS_TO_READ_EXTENDED = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16)

# Time series configuration
TIME_SERIES_POINTS = 100  # Number of points for time series plots

# Data type labels for displacement analysis
DISPLACEMENT_DATA_LABELS = [
    'abs total disp', 'density - mass', 'temp (K)', 
    'z disp (A)', 'lateral disp (A)', 'outward disp vector (A)'
]

# Standard axis limits for different plot types
AXIS_LIMITS = {
    'filament_height': {'y': (3, 25)},
    'gap_size': {'y': (0, 350)},
    'default': None,
}


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class PlotConfig:
    """
    Configuration class for plot appearance and styling.
    
    Attributes:
        colors: List of colors for multi-series plots
        linestyles: List of line styles for multi-series plots  
        markers: List of markers for multi-series plots
        font_sizes: Dictionary of font sizes for different elements
        defaults: Dictionary of default plot parameters
        layout: Dictionary of layout parameters
        output_format: Default output format for saved figures
    """
    colors: List[str] = None
    linestyles: List[str] = None
    markers: List[str] = None
    font_sizes: Dict[str, int] = None
    defaults: Dict[str, Any] = None
    layout: Dict[str, Any] = None
    output_format: str = DEFAULT_OUTPUT_FORMAT
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.colors is None:
            self.colors = DEFAULT_COLORS.copy()
        if self.linestyles is None:
            self.linestyles = DEFAULT_LINESTYLES.copy()
        if self.markers is None:
            self.markers = DEFAULT_MARKERS.copy()
        if self.font_sizes is None:
            self.font_sizes = FONT_SIZES.copy()
        if self.defaults is None:
            self.defaults = PLOT_DEFAULTS.copy()
        if self.layout is None:
            self.layout = LAYOUT_DEFAULTS.copy()


@dataclass
class TimeSeriesConfig:
    """
    Configuration class for time series plotting.
    
    Attributes:
        ncolumns: Number of columns in subplot grid
        figsize_multiplier: Multiplier for calculating figure size
        time_points: Number of time points to plot
        data_labels: Labels for different data types
    """
    ncolumns: int = 4
    figsize_multiplier: Tuple[float, float] = (3.0, 0.65)  # (width_per_col, height_per_row)
    time_points: int = TIME_SERIES_POINTS
    data_labels: List[str] = None
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.data_labels is None:
            self.data_labels = DISPLACEMENT_DATA_LABELS.copy()
    
    def calculate_figsize(self, nrows: int) -> Tuple[float, float]:
        """
        Calculate figure size based on number of rows and columns.
        
        Args:
            nrows: Number of rows in the subplot grid
            
        Returns:
            Tuple of (width, height) for the figure
        """
        width = self.ncolumns * self.figsize_multiplier[0]
        height = nrows * self.figsize_multiplier[1]
        return (width, height)


@dataclass
class DataConfig:
    """
    Configuration class for data processing settings.
    
    Attributes:
        columns_default: Default columns to read from data files
        columns_extended: Extended set of columns for detailed analysis
        validation_enabled: Whether to enable data validation
    """
    columns_default: Tuple[int, ...] = COLUMNS_TO_READ_DEFAULT
    columns_extended: Tuple[int, ...] = COLUMNS_TO_READ_EXTENDED
    validation_enabled: bool = True


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

# Global default configurations that can be imported and used
DEFAULT_PLOT_CONFIG = PlotConfig()
DEFAULT_TIMESERIES_CONFIG = TimeSeriesConfig()
DEFAULT_DATA_CONFIG = DataConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_axis_limits(plot_type: str, axis: str = 'y') -> Optional[Tuple[float, float]]:
    """
    Get standard axis limits for a given plot type.
    
    Args:
        plot_type: Type of plot (e.g., 'filament_height', 'gap_size')
        axis: Which axis ('x' or 'y')
        
    Returns:
        Tuple of (min, max) limits, or None if no standard limits defined
    """
    if plot_type in AXIS_LIMITS and axis in AXIS_LIMITS[plot_type]:
        return AXIS_LIMITS[plot_type][axis]
    return AXIS_LIMITS.get('default')


def cycle_plot_style(index: int, config: PlotConfig = None) -> Dict[str, Any]:
    """
    Get plot style parameters cycling through available options.
    
    Args:
        index: Index to determine which style to use
        config: Plot configuration to use (defaults to DEFAULT_PLOT_CONFIG)
        
    Returns:
        Dictionary with plot style parameters
    """
    if config is None:
        config = DEFAULT_PLOT_CONFIG
    
    color_idx = index % len(config.colors)
    style_idx = index % len(config.linestyles)
    marker_idx = index % len(config.markers)
    
    return {
        'color': config.colors[color_idx],
        'linestyle': config.linestyles[style_idx],
        'marker': config.markers[marker_idx],
        'markersize': config.defaults['markersize'],
        'linewidth': config.defaults['linewidth'],
        'alpha': config.defaults['alpha'],
    }


def create_figure_with_subplots(
    nrows: int, 
    ncolumns: int, 
    figsize: Tuple[float, float],
    config: PlotConfig = None
) -> Tuple[Any, Any]:
    """
    Create a figure with subplots using standard configuration.
    
    Args:
        nrows: Number of rows in subplot grid
        ncolumns: Number of columns in subplot grid  
        figsize: Figure size as (width, height)
        config: Plot configuration to use (defaults to DEFAULT_PLOT_CONFIG)
        
    Returns:
        Tuple of (figure, axes) objects
    """
    import matplotlib.pyplot as plt
    
    if config is None:
        config = DEFAULT_PLOT_CONFIG
    
    fig, axes = plt.subplots(
        nrows, ncolumns,
        squeeze=config.layout['squeeze'],
        constrained_layout=config.layout['constrained_layout'],
        figsize=figsize
    )
    
    if config.layout['tight_layout']:
        fig.tight_layout()
        
    return fig, axes


def configure_subplot_appearance(
    ax: Any,
    config: PlotConfig = None,
    hide_xticks: bool = False,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    show_legend: bool = False
) -> None:
    """
    Configure subplot appearance with standard settings.
    
    Args:
        ax: Matplotlib axes object
        config: Plot configuration to use (defaults to DEFAULT_PLOT_CONFIG)
        hide_xticks: Whether to hide x-tick labels
        ylabel: Y-axis label
        xlabel: X-axis label  
        show_legend: Whether to show the legend
    """
    if config is None:
        config = DEFAULT_PLOT_CONFIG
    
    # Standard subplot configuration
    ax.adjustable = 'datalim'
    ax.set_aspect('auto')
    ax.tick_params(
        axis='both', which='major',
        labelsize=config.font_sizes['tick_major']
    )
    
    # Hide x-tick labels if requested
    if hide_xticks:
        ax.set_xticklabels(())
    
    # Set labels if provided
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=config.font_sizes['label_small'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=config.font_sizes['label_large'])
    
    # Configure legend if requested
    if show_legend:
        ax.legend(
            loc=config.defaults['legend_location'],
            fontsize=config.font_sizes['label_medium']
        )


def save_figure(
    fig: Any,
    output_dir: str,
    filename: str,
    config: PlotConfig = None,
    close_after_save: bool = True
) -> str:
    """
    Save figure with standard configuration.
    
    Args:
        fig: Matplotlib figure object
        output_dir: Directory to save the figure
        filename: Base filename (without extension)
        config: Plot configuration to use (defaults to DEFAULT_PLOT_CONFIG)  
        close_after_save: Whether to close the figure after saving
        
    Returns:
        Path to the saved file
    """
    import os
    import matplotlib.pyplot as plt
    
    if config is None:
        config = DEFAULT_PLOT_CONFIG
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply subplot adjustments if configured
    if config.layout['subplot_adjust']:
        plt.subplots_adjust(**config.layout['subplot_adjust'])
    
    # Save figure
    savepath = os.path.join(output_dir, f"{filename}.{config.output_format}")
    fig.savefig(
        savepath,
        bbox_inches=config.defaults['bbox_inches'],
        format=config.output_format
    )
    
    if close_after_save:
        plt.close(fig)
        
    return savepath


def process_displacement_timeseries_data(
    file_list: List[str],
    loop_start: int,
    loop_end: int,
    time_points: int = 100,
    read_displacement_data_func=None
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Process displacement data from multiple files for time series plotting.
    
    Args:
        file_list: List of file paths containing displacement data
        loop_start: Starting timestep for data processing
        loop_end: Ending timestep for data processing
        time_points: Number of time points for plotting
        read_displacement_data_func: Function to read displacement data
        
    Returns:
        Tuple of (all_thermo_data, element_labels, dump_steps)
    """
    if read_displacement_data_func is None:
        raise ValueError("read_displacement_data_func must be provided")
    
    # Read and process data from all files
    all_thermo_data = []
    element_labels = []
    
    for filename in file_list:
        # Extract element label from filename (first 2 characters)
        element_labels.append(filename[:2])
        all_thermo_data.append(read_displacement_data_func(filename, loop_start, loop_end))
    
    # Convert to numpy array for easier indexing
    all_thermo_data = np.array(all_thermo_data)
    
    # Generate time steps for plotting
    dump_steps = np.linspace(0, loop_end - 1, time_points)
    
    return all_thermo_data, element_labels, dump_steps


def plot_timeseries_grid(
    data: np.ndarray,
    x_values: np.ndarray,
    element_labels: List[str],
    datatype: str,
    data_labels: List[str],
    dataindex: int,
    nrows: int,
    ncolumns: int,
    figsize: Tuple[float, float],
    config: PlotConfig = None
) -> Any:
    """
    Create a grid of time series plots.
    
    Args:
        data: 4D array [file_index, timestep, bin_number, data_type]
        x_values: Array of x-axis values (time steps)
        element_labels: Labels for each file/element
        datatype: Type of data being plotted (for labeling)
        data_labels: Labels for different data types
        dataindex: Index into data array specifying which data type to plot
        nrows: Number of rows in subplot grid
        ncolumns: Number of columns in subplot grid
        figsize: Figure size as (width, height)
        config: Plot configuration to use
        
    Returns:
        Matplotlib figure object
    """
    if config is None:
        config = DEFAULT_PLOT_CONFIG
    
    # Create figure and axes
    fig, axes = create_figure_with_subplots(nrows, ncolumns, figsize, config)
    
    # Plot data for each file (column) and each spatial bin (row)
    for j in range(ncolumns):
        for i in range(nrows):
            # Plot timeseries for this file and spatial bin
            # Note: plotting from bottom to top (nrows-1-i) for spatial ordering
            axes[nrows-1-i, j].plot(
                x_values,
                data[j, :, i, dataindex],
                label=f'{element_labels[j]} of region {i+1}',
                color='blue'
            )
            
            # Configure subplot appearance
            ylabel = f'{datatype} \n {data_labels[dataindex]}' if j == 0 else None
            xlabel = 'Dump steps' if nrows-1-i == nrows-1 else None
            hide_xticks = nrows-1-i != nrows-1
            
            configure_subplot_appearance(
                axes[nrows-1-i, j],
                config=config,
                hide_xticks=hide_xticks,
                ylabel=ylabel,
                xlabel=xlabel,
                show_legend=True
            )
    
    return fig
