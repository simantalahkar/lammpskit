"""
Plotting utilities specific to electrochemical cell model analysis.

This module contains plotting functions for HfTaO simulations including
time series plots, displacement visualizations, and ecellmodel-specific
plotting configurations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Callable
from ..config import PlotConfig, DEFAULT_PLOT_CONFIG


def process_displacement_timeseries_data(
    file_list: List[str],
    start_time: int,
    end_time: int, 
    time_points: int,
    read_displacement_data_func: Callable
) -> tuple[List[List[np.ndarray]], List[str], np.ndarray]:
    """
    Process displacement time series data from multiple files.
    
    This function reads displacement data from multiple files and processes
    it for time series plotting in electrochemical cell simulations.
    
    Parameters
    ----------
    file_list : List[str]
        List of file paths to displacement data files.
    start_time : int
        Starting time for data reading.
    end_time : int
        Ending time for data reading.
    time_points : int
        Number of time points for the analysis.
    read_displacement_data_func : Callable
        Function to read displacement data from files.
        
    Returns
    -------
    tuple
        Tuple containing:
        - all_thermo_data: List of displacement data arrays
        - element_labels: List of element labels extracted from filenames
        - dump_steps: Array of time steps
    """
    # Validate inputs
    if read_displacement_data_func is None:
        raise ValueError("read_displacement_data_func must be provided")
    
    # Validate time_points
    if not isinstance(time_points, int) or time_points <= 0:
        raise ValueError("time_points must be a positive integer")
    
    # Call validation functions from config
    from ..config import validate_file_list, validate_loop_parameters
    validate_file_list(file_list)
    validate_loop_parameters(start_time, end_time)
    
    from .data_processing import extract_element_label_from_filename
    
    # Read thermodynamic data for each file
    all_thermo_data: List[List[np.ndarray]] = []
    element_labels: List[str] = []
    
    for filename in file_list:
        # Extract element label from filename
        element_label = extract_element_label_from_filename(filename)
        element_labels.append(element_label)
        
        # Read displacement data with error handling
        try:
            thermo_data = read_displacement_data_func(filename, start_time, end_time)
            all_thermo_data.append(thermo_data)
        except Exception as e:
            raise ValueError(f"Failed to process file {filename}: {str(e)}")
    
    # Create dump steps array
    dump_steps = np.arange(start_time, end_time + 1)
    
    return all_thermo_data, element_labels, dump_steps


def plot_timeseries_grid(
    data: List[List[np.ndarray]],
    x_values: np.ndarray,
    element_labels: List[str],
    datatype: str,
    data_labels: List[str],
    dataindex: int,
    nrows: int,
    ncolumns: int,
    figsize: tuple,
    config: PlotConfig = None
) -> plt.Figure:
    """
    Create a grid of time series subplots for displacement data.
    
    This function creates a grid where each row represents a spatial bin
    and each column represents a different element/file. Used specifically
    for electrochemical cell displacement analysis.
    
    Parameters
    ----------
    data : List[List[np.ndarray]]
        Displacement data arrays organized by file and time.
    x_values : np.ndarray
        X-axis values (typically time steps).
    element_labels : List[str]
        Labels for each element/file.
    datatype : str
        Type of data being plotted.
    data_labels : List[str]
        Labels for different data types.
    dataindex : int
        Index of the specific data type to plot.
    nrows : int
        Number of subplot rows.
    ncolumns : int
        Number of subplot columns.
    figsize : tuple
        Figure size (width, height).
    config : PlotConfig, optional
        Plot configuration object.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    if config is None:
        config = DEFAULT_PLOT_CONFIG
    
    # Create figure and subplots
    plt.ioff()
    fig, axes = plt.subplots(nrows, ncolumns, figsize=figsize, squeeze=False)
    
    # Convert data to numpy array for easier indexing
    data_array = np.array(data)
    
    # Plot data for each subplot
    for row in range(nrows):
        for col in range(min(ncolumns, len(element_labels))):
            ax = axes[row, col]
            
            # Extract data for this bin and element
            if row < data_array.shape[1] and col < data_array.shape[0]:
                # Reverse the data order: bottom row gets highest data index, top row gets lowest
                data_row_index = nrows - 1 - row
                y_data = data_array[col, :, data_row_index, dataindex]
                
                # Create legend label with 1-based chunk numbering (bottom=1, top=highest)
                chunk_id = data_row_index + 1
                legend_label = f"{element_labels[col]} in Chunk {chunk_id}"
                ax.plot(x_values, y_data, 
                       linewidth=config.linewidth,
                       alpha=config.alpha,
                       color=config.colors[col % len(config.colors)],
                       label=legend_label)
                
                # Configure subplot appearance
                if row == 0:  # Top row gets column titles
                    ax.set_title(f"{element_labels[col]}", fontsize=config.title_fontsize)
                
                if row == nrows - 1:  # Bottom row gets x-label
                    ax.set_xlabel("Time step", fontsize=config.label_fontsize)
                
                # Set minimal tick labels for scale reference
                # Use default matplotlib tick behavior for better appearance
                ax.tick_params(labelsize=config.tick_fontsize)
                
                # Add legend to all subplots
                ax.legend(fontsize=config.label_fontsize, loc='best')
                
                # Add grid if enabled
                if config.grid:
                    ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for col in range(len(element_labels), ncolumns):
        for row in range(nrows):
            fig.delaxes(axes[row, col])
    
    # Add shared y-label for the leftmost column
    # Position it much closer to the y-axes (half the previous distance)
    shared_ylabel = f"{datatype} {data_labels[dataindex]}"
    fig.text(0.025, 0.5, shared_ylabel, fontsize=config.label_fontsize, 
             rotation=90, va='center', ha='center')
    
    # Set overall title
    fig.suptitle(f'{datatype} {data_labels[dataindex]}', fontsize=config.title_fontsize)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.05)  # Reduced left margin for closer positioning
    
    return fig


def create_and_save_figure(
    plot_func: Callable,
    data: Any,
    output_filename: str,
    output_dir: str = os.getcwd(),
    config: PlotConfig = None,
    **kwargs
) -> plt.Figure:
    """
    Create and save a figure using a specified plotting function.
    
    This is a utility function for standardizing figure creation and saving
    across different plot types in electrochemical cell analysis.
    
    Parameters
    ----------
    plot_func : Callable
        Function to create the plot.
    data : Any
        Data to be plotted.
    output_filename : str
        Base filename for saving (without extension).
    output_dir : str, optional
        Directory to save the figure.
    config : PlotConfig, optional
        Plot configuration object.
    **kwargs
        Additional keyword arguments for the plotting function.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    if config is None:
        config = DEFAULT_PLOT_CONFIG
    
    # Create the figure using the provided function
    fig = plot_func(data, **kwargs)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in multiple formats if specified
    for fmt in config.save_formats:
        filename = f"{output_filename}.{fmt}"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, 
                   dpi=config.dpi,
                   bbox_inches='tight',
                   format=fmt)
    
    return fig
