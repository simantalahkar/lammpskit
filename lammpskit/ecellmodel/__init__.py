"""
Electrochemical Cell Model Analysis Package

This package provides specialized analysis tools for HfTaO electrochemical
cell simulations including data processing, plotting, analysis, and workflows.
"""

# Import specialized ecellmodel modules
from . import data_processing
from . import plotting  
from . import analysis
from . import workflows

# Import main analysis function for convenience
from .workflows import run_complete_analysis

# Import main simulation analysis
from .filament_layer_analysis import main

__all__ = [
    'data_processing',
    'plotting', 
    'analysis',
    'workflows',
    'run_complete_analysis',
    'main'
]