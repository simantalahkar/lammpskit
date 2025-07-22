"""
LAMMPSKit: A modular toolkit for LAMMPS molecular dynamics simulation analysis.

This package provides:
- General I/O utilities for reading LAMMPS trajectory files
- General plotting utilities for scientific visualization
- Specialized analysis modules for specific simulation types
  - ecellmodel: Electrochemical cell simulation analysis
"""

# Version info
__version__ = "0.2.2"

# Import key modules for convenient access
from . import io
from . import plotting
from . import config
from . import ecellmodel

__all__ = [
    'io',
    'plotting', 
    'config',
    'ecellmodel',
]