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
import os


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
TIME_SERIES_POINTS = 100  # Number of points for resolution of time series plots

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
    
    # Convenience properties for easy access to commonly used values
    @property
    def linewidth(self):
        """Get default line width."""
        return self.defaults.get('linewidth', 1.2)
    
    @property
    def alpha(self):
        """Get default alpha value."""
        return self.defaults.get('alpha', 0.8)
    
    @property
    def figsize(self):
        """Get default figure size."""
        return self.defaults.get('figsize', (10, 6))
    
    @property
    def dpi(self):
        """Get default DPI for saving figures."""
        return self.defaults.get('dpi', 300)
    
    @property
    def save_formats(self):
        """Get list of save formats."""
        return self.defaults.get('save_formats', ['png'])
    
    @property
    def grid(self):
        """Get grid setting."""
        return self.defaults.get('grid', True)
    
    @property
    def title_fontsize(self):
        """Get title font size."""
        return self.font_sizes.get('title', 12)
    
    @property
    def label_fontsize(self):
        """Get label font size."""
        return self.font_sizes.get('label', 10)
    
    @property
    def tick_fontsize(self):
        """Get tick font size."""
        return self.font_sizes.get('tick', 8)


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
# VALIDATION FUNCTIONS
# =============================================================================

def validate_dataindex(dataindex: int, max_index: Optional[int] = None) -> None:
    """
    Validate that dataindex is within the valid range for displacement data.
    
    Args:
        dataindex: Index to validate (supports negative indexing like Python lists)
        max_index: Maximum valid index (defaults to length of DISPLACEMENT_DATA_LABELS - 1)
        
    Raises:
        ValueError: If dataindex is out of valid range
    """
    if max_index is None:
        max_index = len(DISPLACEMENT_DATA_LABELS) - 1
    
    if not isinstance(dataindex, int):
        raise ValueError(f"dataindex must be an integer, got {type(dataindex).__name__}")
    
    # Support Python-style negative indexing
    total_length = max_index + 1
    
    # Check bounds (both positive and negative indexing)
    if dataindex < -total_length or dataindex > max_index:
        raise ValueError(
            f"dataindex {dataindex} is out of range. Valid range is {-total_length}-{max_index}. "
            f"Available data types: {DISPLACEMENT_DATA_LABELS}"
        )


def validate_file_list(file_list: List[str]) -> None:
    """
    Validate that all files in the file list exist and are readable.
    
    Args:
        file_list: List of file paths to validate
        
    Raises:
        FileNotFoundError: If any file does not exist
        ValueError: If file_list is empty
    """
    if not file_list:
        raise ValueError("file_list cannot be empty")
    
    if not isinstance(file_list, (list, tuple)):
        raise ValueError(f"file_list must be a list or tuple, got {type(file_list).__name__}")
    
    missing_files = []
    for filepath in file_list:
        if not isinstance(filepath, str):
            raise ValueError(f"All file paths must be strings, got {type(filepath).__name__}")
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        raise FileNotFoundError(f"The following files were not found: {missing_files}")


def validate_loop_parameters(loop_start: int, loop_end: int) -> None:
    """
    Validate loop start and end parameters.
    
    Args:
        loop_start: Starting loop index
        loop_end: Ending loop index
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(loop_start, int) or not isinstance(loop_end, int):
        raise ValueError("loop_start and loop_end must be integers")
    
    if loop_start < 0:
        raise ValueError(f"loop_start must be non-negative, got {loop_start}")
    
    if loop_end < 0:
        raise ValueError(f"loop_end must be non-negative, got {loop_end}")
        
    if loop_start > loop_end:
        raise ValueError(
            f"loop_start ({loop_start}) must be less than or equal to loop_end ({loop_end})"
        )


def validate_chunks_parameter(nchunks: int, min_chunks: int = 1, max_chunks: int = 1000) -> None:
    """
    Validate the number of spatial chunks parameter.
    
    Args:
        nchunks: Number of spatial chunks
        min_chunks: Minimum allowed chunks (default: 1)
        max_chunks: Maximum allowed chunks (default: 1000)
        
    Raises:
        ValueError: If nchunks is invalid
    """
    if not isinstance(nchunks, int):
        raise ValueError(f"nchunks must be an integer, got {type(nchunks).__name__}")
    
    if nchunks < min_chunks:
        raise ValueError(f"nchunks must be at least {min_chunks}, got {nchunks}")
        
    if nchunks > max_chunks:
        raise ValueError(f"nchunks cannot exceed {max_chunks}, got {nchunks}")


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


# =============================================================================
# DEFAULT CONFIGURATION INSTANCES
# =============================================================================

# Create default configuration instances for easy import and use
DEFAULT_PLOT_CONFIG = PlotConfig()
DEFAULT_TIMESERIES_CONFIG = TimeSeriesConfig()
DEFAULT_DATA_CONFIG = DataConfig()
