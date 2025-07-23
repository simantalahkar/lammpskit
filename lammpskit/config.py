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


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

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


# Legacy alias for backward compatibility (in case any tests use it)
extract_element_label_from_filename = None  # Will be imported from data_processing module
