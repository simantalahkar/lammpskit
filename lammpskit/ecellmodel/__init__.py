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

__all__ = [
    'data_processing',
    'main'
]