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
    """Configuration for simple time series plots."""
    
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
    
    # Legend and text
    fontsize_legend: Optional[int] = None  # None means use default


@dataclass
class DualAxisPlotConfig:
    """Configuration for dual-axis plots with two y-axes."""
    
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


def create_time_series_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    stats_label: str,
    config: Optional[TimeSeriesPlotConfig] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend_fontsize: Optional[int] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a standardized time series plot with configurable line and scatter elements.
    
    This function provides flexibility to create either line plots, scatter plots,
    or combined line+scatter plots based on the configuration.
    
    Parameters
    ----------
    x_data : np.ndarray
        X-axis data (typically time).
    y_data : np.ndarray
        Y-axis data to plot.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    stats_label : str
        Label showing statistics or other information for legend.
    config : TimeSeriesPlotConfig, optional
        Plot configuration. Uses defaults if None.
    ylim : tuple of float, optional
        Y-axis limits (min, max).
    legend_fontsize : int, optional
        Override legend font size from config.
        
    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : plt.Axes
        Axes object.
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
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set y-limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add legend (use parameter override, then config, then matplotlib default)
    if config.include_scatter:  # Only show legend if there's a label
        fontsize = legend_fontsize or config.fontsize_legend
        if fontsize is not None:
            ax.legend(fontsize=fontsize)
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
    secondary_ylim: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Create a standardized dual-axis plot with two y-axes.
    
    Parameters
    ----------
    x_data : np.ndarray
        X-axis data (typically time).
    primary_y_data : np.ndarray
        Primary y-axis data (left side).
    secondary_y_data : np.ndarray
        Secondary y-axis data (right side).
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    primary_ylabel : str
        Primary y-axis label.
    secondary_ylabel : str
        Secondary y-axis label.
    primary_stats_label : str
        Label for primary data with statistics.
    secondary_stats_label : str
        Label for secondary data with statistics.
    config : DualAxisPlotConfig, optional
        Plot configuration. Uses defaults if None.
    primary_ylim : tuple of float, optional
        Primary y-axis limits (min, max).
    secondary_ylim : tuple of float, optional
        Secondary y-axis limits (min, max).
        
    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax1 : plt.Axes
        Primary axes object (left y-axis).
    ax2 : plt.Axes
        Secondary axes object (right y-axis).
    """
    if config is None:
        config = DualAxisPlotConfig()
    
    fig, ax1 = plt.subplots()
    
    # Configure primary axis (left)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(primary_ylabel, color=config.primary_color)
    ax1.scatter(x_data, primary_y_data, 
               alpha=config.alpha, 
               linewidth=config.linewidth, 
               s=config.markersize, 
               color=config.primary_color, 
               label=primary_stats_label)
    ax1.tick_params(axis='y', labelcolor=config.primary_color)
    
    # Set primary y-limits if provided
    if primary_ylim is not None:
        ax1.set_ylim(primary_ylim)
    
    # Create secondary axis (right)
    ax2 = ax1.twinx()
    ax2.set_ylabel(secondary_ylabel, color=config.secondary_color)
    ax2.scatter(x_data, secondary_y_data, 
               alpha=config.alpha, 
               linewidth=config.linewidth, 
               s=config.markersize, 
               marker=config.marker, 
               color=config.secondary_color, 
               label=secondary_stats_label)
    ax2.tick_params(axis='y', labelcolor=config.secondary_color)
    
    # Set secondary y-limits if provided
    if secondary_ylim is not None:
        ax2.set_ylim(secondary_ylim)
    
    # Set title
    plt.title(title)
    
    # Apply tight layout if configured
    if config.tight_layout:
        fig.tight_layout()
    
    # Add legends
    ax1.legend(loc=config.primary_legend_loc, framealpha=config.legend_framealpha)
    ax2.legend(loc=config.secondary_legend_loc, framealpha=config.legend_framealpha)
    
    return fig, ax1, ax2


def save_and_close_figure(
    fig: plt.Figure,
    output_dir: str,
    filename: str,
    file_format: str = 'pdf'
) -> None:
    """
    Save a figure to disk and close it to free memory.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure object to save.
    output_dir : str
        Directory to save the figure.
    filename : str
        Base filename (without extension).
    file_format : str, optional
        File format for saving (default: 'pdf').
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{filename}.{file_format}"
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close(fig)


def calculate_mean_std_label(data: np.ndarray, label_prefix: str, precision: int = 2) -> str:
    """
    Calculate mean and standard deviation and format as a label string.
    
    Parameters
    ----------
    data : np.ndarray
        Data array to calculate statistics for.
    label_prefix : str
        Prefix text for the label.
    precision : int, optional
        Decimal precision for formatting (default: 2).
        
    Returns
    -------
    str
        Formatted statistics label: "prefix = mean +/- std".
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    return f"{label_prefix} = {mean_val:.{precision}f} +/- {std_val:.{precision}f}"


def calculate_frequency_label(data: np.ndarray, target_value, label_template: str, precision: int = 2) -> str:
    """
    Calculate frequency of a target value and format as a label string.
    
    Parameters
    ----------
    data : np.ndarray
        Data array to calculate frequency for.
    target_value : 
        Value to count frequency of.
    label_template : str
        Template string with {frequency} placeholder.
    precision : int, optional
        Decimal precision for formatting (default: 2).
        
    Returns
    -------
    str
        Formatted frequency label.
        
    Examples
    --------
    >>> data = np.array([1, 0, 1, 1, 0])
    >>> calculate_frequency_label(data, 1, "connected {frequency:.1f}% of time")
    'connected 60.0% of time'
    """
    frequency = np.sum(data == target_value) / len(data) * 100
    return label_template.format(frequency=frequency)
