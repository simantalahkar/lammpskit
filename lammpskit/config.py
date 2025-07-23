"""
LAMMPSKit Configuration Module - Simplified
==========================================

Essential validation functions and constants for LAMMPSKit package.
This simplified version removes over-engineered configuration classes
and keeps only the functions and constants actually used by multiple modules.
"""

from typing import List, Optional
import os


# =============================================================================
# CONSTANTS USED BY MULTIPLE MODULES
# =============================================================================

# Data type labels for displacement analysis (used by tests)
DISPLACEMENT_DATA_LABELS = [
    'abs total disp', 'density - mass', 'temp (K)', 
    'z disp (A)', 'lateral disp (A)', 'outward disp vector (A)'
]

# Default columns to read from LAMMPS coordinate files
# Format: (id, type, charge, x, y, z, ...)
DEFAULT_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)  # Standard analysis columns
EXTENDED_COLUMNS_TO_READ = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16)  # Extended analysis columns


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_filepath(filepath: str, check_existence: bool = True) -> None:
    """
    Validate that filepath is a valid string path.
    
    Args:
        filepath: The file path to validate
        check_existence: Whether to check if file exists
        
    Raises:
        TypeError: If filepath is not a string
        ValueError: If filepath is empty
        FileNotFoundError: If file doesn't exist (when check_existence=True)
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")
    
    if not filepath:
        raise ValueError("filepath cannot be empty")
    
    if check_existence and not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")


def validate_dataindex(dataindex: int, max_index: Optional[int] = None) -> None:
    """
    Validate that dataindex is a valid integer for array indexing.
    
    Args:
        dataindex: The index to validate
        max_index: Optional maximum allowed index
        
    Raises:
        ValueError: If dataindex is invalid
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
    Validate that file_list contains valid file paths.
    
    Args:
        file_list: List of file paths to validate
        
    Raises:
        ValueError: If file_list is invalid
        FileNotFoundError: If any file doesn't exist
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
        raise ValueError("loop_start must be non-negative")
    
    if loop_end < 0:
        raise ValueError("loop_end must be non-negative")
    
    if loop_start > loop_end:
        raise ValueError(f"loop_start ({loop_start}) must be less than or equal to loop_end ({loop_end})")


def validate_chunks_parameter(nchunks: int, min_chunks: int = 1, max_chunks: int = 1000) -> None:
    """
    Validate chunks parameter for spatial binning.
    
    Args:
        nchunks: Number of chunks/bins
        min_chunks: Minimum allowed chunks
        max_chunks: Maximum allowed chunks
        
    Raises:
        ValueError: If nchunks is invalid
    """
    if not isinstance(nchunks, int):
        raise ValueError("nchunks must be an integer")
    
    # Provide more specific error messages like the original
    if nchunks < min_chunks:
        raise ValueError(f"nchunks must be at least {min_chunks}")
    
    if nchunks > max_chunks:
        raise ValueError(f"nchunks cannot exceed {max_chunks}")


def validate_cluster_parameters(
    z_filament_lower_limit: float,
    z_filament_upper_limit: float, 
    thickness: float
) -> None:
    """
    Validate cluster analysis parameters for OVITO processing.
    
    Args:
        z_filament_lower_limit: Lower z-bound for filament connection
        z_filament_upper_limit: Upper z-bound for filament connection  
        thickness: Filament thickness parameter
        
    Raises:
        TypeError: If parameters are not numeric
        ValueError: If parameter ranges are invalid
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
        raise ValueError(f"z_filament_lower_limit ({z_filament_lower_limit}) must be less than z_filament_upper_limit ({z_filament_upper_limit})")
    if thickness <= 0:
        raise ValueError(f"thickness ({thickness}) must be positive")
    
    # Parameter range validation (warnings)
    if z_filament_lower_limit < 0:
        warnings.warn(f"z_filament_lower_limit ({z_filament_lower_limit}) is negative, which might indicate coordinate system issues", UserWarning)
    
    if abs(z_filament_lower_limit) > 1000 or abs(z_filament_upper_limit) > 1000:
        warnings.warn(f"Large z-coordinate values detected (z_lower={z_filament_lower_limit}, z_upper={z_filament_upper_limit}), which might indicate unit scale issues", UserWarning)


# Legacy alias for backward compatibility (in case any tests use it)
extract_element_label_from_filename = None  # Will be imported from data_processing module
