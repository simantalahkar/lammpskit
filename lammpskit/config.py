"""
LAMMPSKit Configuration Module
==============================

Essential validation functions and constants for robust LAMMPS trajectory analysis.
This module provides infrastructure for input validation, parameter checking, and
standardized constants used across LAMMPSKit analysis workflows.

Architecture Design
-------------------
The configuration module follows a functional approach rather than class-based
configuration objects, prioritizing simplicity and direct validation at point-of-use.
This design reduces coupling while ensuring consistent parameter validation across
all analysis modules.

Key Components
--------------
- Column mapping constants for LAMMPS dump file parsing
- Data type labels for displacement and property analysis
- Input validation functions with domain-specific error messages
- Parameter range checking with physics-aware warnings

Validation Philosophy
---------------------
Validation functions use a "fail-fast" approach with informative error messages
to catch configuration issues early in analysis workflows. Physics-aware warnings
help identify potential coordinate system or unit scale problems common in MD
simulations.

Performance Considerations
--------------------------
Validation overhead is O(1) for most functions, O(n) for file list validation.
Pre-validate parameters once at workflow start rather than per-timestep for
optimal performance in long trajectory analysis.
"""

from typing import List, Optional
import os


# =============================================================================
# LAMMPS DATA STRUCTURE CONSTANTS
# =============================================================================

# Data type labels for displacement analysis and statistical reporting
# Used by plotting functions and test validation to ensure consistent labeling
DISPLACEMENT_DATA_LABELS = [
    "abs total disp",  # Total displacement magnitude (Angstroms)
    "density - mass",  # Mass density (g/cm³)
    "temp (K)",  # Temperature (Kelvin)
    "z disp (A)",  # Vertical displacement component (Angstroms)
    "lateral disp (A)",  # Horizontal displacement magnitude (Angstroms)
    "outward disp vector (A)",  # Radial displacement component (Angstroms)
]

# LAMMPS dump file column mappings for trajectory parsing
# Standard format: id type charge x y z vx vy vz fx fy fz
# Indices correspond to: (id, type, charge, x, y, z, vx, vy, vz, fx, fy, fz)
DEFAULT_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)  # Core analysis columns
EXTENDED_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16)  # Full property set


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_filepath(filepath: str, check_existence: bool = True) -> None:
    """
    Validate file path for LAMMPS trajectory and output files.

    Ensures filepath is a valid string and optionally verifies file existence.
    Essential for preventing downstream failures in trajectory reading and
    analysis output generation.

    Parameters
    ----------
    filepath : str
        Path to file for validation. Supports both absolute and relative paths.
    check_existence : bool, optional
        Whether to verify file exists on disk (default: True).
        Set False for output file validation.

    Raises
    ------
    TypeError
        If filepath is not a string.
    ValueError
        If filepath is empty string.
    FileNotFoundError
        If file doesn't exist and check_existence=True.

    Examples
    --------
    Validate input trajectory file:

    >>> validate_filepath('trajectory.lammpstrj')

    Validate output path without existence check:

    >>> validate_filepath('output/analysis.pdf', check_existence=False)
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    if not filepath:
        raise ValueError("filepath cannot be empty")

    if check_existence and not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")


def validate_dataindex(dataindex: int, max_index: Optional[int] = None) -> None:
    """
    Validate array index for displacement data and property arrays.

    Ensures safe array indexing with support for Python negative indexing.
    Primarily used for accessing DISPLACEMENT_DATA_LABELS and trajectory
    property arrays.

    Parameters
    ----------
    dataindex : int
        Index to validate. Supports negative indexing (e.g., -1 for last element).
    max_index : int, optional
        Maximum allowed index. If None, uses DISPLACEMENT_DATA_LABELS length.

    Raises
    ------
    ValueError
        If dataindex is not integer or out of valid range.

    Notes
    -----
    Negative indexing follows Python conventions: -1 = last element,
    -n = first element for array of length n.

    Examples
    --------
    Validate index for displacement data:

    >>> validate_dataindex(2)  # Access 'temp (K)'
    >>> validate_dataindex(-1)  # Access last element
    """
    if not isinstance(dataindex, int):
        raise ValueError("dataindex must be an integer")

    # Allow negative indexing (common in Python)
    if max_index is not None:
        if dataindex >= max_index or dataindex < -max_index:
            raise ValueError(f"dataindex {dataindex} is out of range for max_index {max_index}")
    else:
        # Use the DISPLACEMENT_DATA_LABELS length for validation like the original
        total_length = len(DISPLACEMENT_DATA_LABELS)
        if dataindex < -total_length or dataindex >= total_length:
            raise ValueError(f"dataindex {dataindex} is out of range")


def validate_file_list(file_list: List[str]) -> None:
    """
    Validate list of trajectory files for batch processing.

    Ensures all files exist and are accessible before starting computationally
    expensive analysis workflows. Prevents partial analysis completion due to
    missing files in multi-trajectory studies.

    Parameters
    ----------
    file_list : List[str]
        List of file paths to validate. Commonly used for time series analysis
        across multiple LAMMPS dump files.

    Raises
    ------
    ValueError
        If file_list is not a list/tuple, is empty, or contains non-string elements.
    FileNotFoundError
        If any files don't exist. Reports all missing files simultaneously
        for efficient error handling.

    Performance Notes
    -----------------
    Complexity: O(n) where n is number of files.
    For large file lists (>1000), consider validating in chunks.

    Examples
    --------
    Validate trajectory sequence:

    >>> files = ['step_0.lammpstrj', 'step_1000.lammpstrj', 'step_2000.lammpstrj']
    >>> validate_file_list(files)
    """
    if not isinstance(file_list, (list, tuple)):
        raise ValueError("file_list must be a list")

    if not file_list:
        raise ValueError("file_list cannot be empty")

    # Collect all missing files to report them together (like original)
    missing_files = []
    for filepath in file_list:
        if not isinstance(filepath, str):
            raise ValueError("All file paths must be strings")
        if not os.path.exists(filepath):
            missing_files.append(filepath)

    if missing_files:
        raise FileNotFoundError(f"The following files were not found: {missing_files}")


def validate_loop_parameters(loop_start: int, loop_end: int) -> None:
    """
    Validate timestep range parameters for trajectory analysis loops.

    Ensures valid timestep iteration bounds for LAMMPS trajectory processing.
    Critical for preventing infinite loops or invalid memory access in
    temporal analysis functions.

    Parameters
    ----------
    loop_start : int
        Starting timestep index (inclusive). Must be non-negative.
    loop_end : int
        Ending timestep index (inclusive). Must be >= loop_start.

    Raises
    ------
    ValueError
        If parameters are not integers, negative, or loop_start > loop_end.

    Notes
    -----
    Both indices are inclusive: range [loop_start, loop_end].
    For trajectory with N timesteps, valid range is [0, N-1].

    Examples
    --------
    Validate analysis range:

    >>> validate_loop_parameters(0, 1000)  # Analyze first 1001 timesteps
    >>> validate_loop_parameters(500, 1500)  # Analyze middle portion
    """
    if not isinstance(loop_start, int) or not isinstance(loop_end, int):
        raise ValueError("loop_start and loop_end must be integers")

    if loop_start < 0:
        raise ValueError("loop_start must be non-negative")

    if loop_end < 0:
        raise ValueError("loop_end must be non-negative")

    if loop_start > loop_end:
        raise ValueError(f"loop_start ({loop_start}) must be less than or equal to loop_end ({loop_end})")


def validate_chunks_parameter(nchunks: int, min_chunks: int = 1, max_chunks: int = 1000) -> None:
    """
    Validate spatial binning parameters for density and distribution analysis.

    Ensures appropriate bin count for spatial discretization in electrochemical
    cell analysis. Balances statistical significance with computational efficiency.

    Parameters
    ----------
    nchunks : int
        Number of spatial bins/chunks for discretization.
    min_chunks : int, optional
        Minimum allowed bins (default: 1). Must be positive.
    max_chunks : int, optional
        Maximum allowed bins (default: 1000). Prevents excessive memory usage.

    Raises
    ------
    ValueError
        If nchunks is not integer or outside [min_chunks, max_chunks] range.

    Performance Notes
    -----------------
    Memory usage: O(nchunks) per property per timestep.
    Computation time: O(N * nchunks) where N is atom count.
    Optimal range: 10-100 chunks for most electrochemical systems.

    Examples
    --------
    Validate binning for layer analysis:

    >>> validate_chunks_parameter(50)  # 50 z-direction layers
    >>> validate_chunks_parameter(10, min_chunks=5, max_chunks=20)
    """
    if not isinstance(nchunks, int):
        raise ValueError("nchunks must be an integer")

    # Provide more specific error messages like the original
    if nchunks < min_chunks:
        raise ValueError(f"nchunks must be at least {min_chunks}")

    if nchunks > max_chunks:
        raise ValueError(f"nchunks cannot exceed {max_chunks}")


def validate_cluster_parameters(z_filament_lower_limit: float, z_filament_upper_limit: float, thickness: float) -> None:
    """
    Validate geometric parameters for filament connectivity analysis.

    Ensures physically meaningful parameters for OVITO-based cluster analysis
    in electrochemical cell simulations. Validates filament detection geometry
    and provides physics-aware warnings for common coordinate system issues.

    Parameters
    ----------
    z_filament_lower_limit : float
        Lower z-coordinate bound for filament connectivity (Angstroms).
        Typically electrode surface position.
    z_filament_upper_limit : float
        Upper z-coordinate bound for filament connectivity (Angstroms).
        Typically opposite electrode surface position.
    thickness : float
        Filament thickness parameter for cluster detection (Angstroms).
        Controls sensitivity of connectivity algorithm.

    Raises
    ------
    TypeError
        If parameters are not numeric (int or float).
    ValueError
        If z_lower >= z_upper or thickness <= 0.

    Warns
    -----
    UserWarning
        If negative z-coordinates detected (potential coordinate system issue).
        If large z-values detected (potential unit scale issue).

    Physics Notes
    -------------
    Typical electrochemical cell dimensions: 20-100 Å electrode separation.
    Filament thickness: 2-10 Å depending on atom size and connectivity criteria.
    Z-coordinates should span electrode-to-electrode distance.

    Examples
    --------
    Validate HfTaO cell parameters:

    >>> validate_cluster_parameters(-10.0, 50.0, 3.5)  # 60 Å cell, 3.5 Å thickness
    >>> validate_cluster_parameters(0.0, 30.0, 2.0)    # 30 Å cell, 2.0 Å thickness
    """
    import warnings

    # Parameter type validation
    if not isinstance(z_filament_lower_limit, (int, float)):
        raise TypeError("z_filament_lower_limit must be numeric (int or float)")
    if not isinstance(z_filament_upper_limit, (int, float)):
        raise TypeError("z_filament_upper_limit must be numeric (int or float)")
    if not isinstance(thickness, (int, float)):
        raise TypeError("thickness must be numeric (int or float)")

    # Parameter range validation (errors)
    if z_filament_lower_limit >= z_filament_upper_limit:
        raise ValueError(
            f"z_filament_lower_limit ({z_filament_lower_limit}) must be less than z_filament_upper_limit ({z_filament_upper_limit})"
        )
    if thickness <= 0:
        raise ValueError(f"thickness ({thickness}) must be positive")

    # Parameter range validation (warnings)
    if z_filament_lower_limit < 0:
        warnings.warn(
            f"z_filament_lower_limit ({z_filament_lower_limit}) is negative, which might indicate coordinate system issues",
            UserWarning,
        )

    if abs(z_filament_lower_limit) > 1000 or abs(z_filament_upper_limit) > 1000:
        warnings.warn(
            f"Large z-coordinate values detected (z_lower={z_filament_lower_limit}, z_upper={z_filament_upper_limit}), which might indicate unit scale issues",
            UserWarning,
        )


# Legacy alias for backward compatibility (in case any tests use it)
extract_element_label_from_filename = None  # Will be imported from data_processing module
