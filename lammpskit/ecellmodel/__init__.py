"""
Electrochemical Cell Model Analysis Package

This package provides specialized analysis tools for HfTaO electrochemical
cell simulations including data processing, plotting, and filament analysis.
"""

# Import specialized ecellmodel modules
from . import data_processing

# plotting module removed - functions inlined into filament_layer_analysis

# Import main simulation analysis
from .filament_layer_analysis import main

# Import core filament analysis functions for direct access
from .filament_layer_analysis import (
    analyze_clusters,
    track_filament_evolution,
    plot_atomic_distribution,
    plot_atomic_charge_distribution,
    plot_displacement_comparison,
    plot_displacement_timeseries,
    read_displacement_data,
)

__all__ = [
    "data_processing",
    "main",
    # Core filament analysis functions
    "analyze_clusters",
    "track_filament_evolution",
    "plot_atomic_distribution",
    "plot_atomic_charge_distribution",
    "plot_displacement_comparison",
    "plot_displacement_timeseries",
    "read_displacement_data",
]
