"""
I/O utilities for LAMMPSKit.

This module provides general-purpose file reading and writing utilities
for LAMMPS simulation data that can be used across different analysis types.
"""

from .lammps_readers import read_structure_info, read_coordinates

__all__ = [
    'read_structure_info',
    'read_coordinates',
]
