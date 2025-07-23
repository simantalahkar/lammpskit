"""
General plotting utilities for LAMMPSKit.

This module provides general-purpose plotting functions that can be used
across different analysis types and simulation workflows.
"""

from .utils import plot_multiple_cases
from .timeseries_plots import (
    TimeSeriesPlotConfig,
    DualAxisPlotConfig,
    create_time_series_plot,
    create_dual_axis_plot,
    save_and_close_figure,
    calculate_mean_std_label,
    calculate_frequency_label
)

__all__ = [
    'plot_multiple_cases',
    'TimeSeriesPlotConfig',
    'DualAxisPlotConfig', 
    'create_time_series_plot',
    'create_dual_axis_plot',
    'save_and_close_figure',
    'calculate_mean_std_label',
    'calculate_frequency_label'
]
