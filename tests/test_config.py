"""
Test suite for validation functions in the LAMMPSKit configuration module.

This module tests the robustness and correctness of input validation functions
that ensure data integrity and provide meaningful error messages.
"""

import pytest
import tempfile
import os

from lammpskit.config import (
    validate_dataindex,
    validate_file_list,
    validate_loop_parameters,
    validate_chunks_parameter,
    DISPLACEMENT_DATA_LABELS
)
from lammpskit.ecellmodel.data_processing import extract_element_label_from_filename


class TestValidateDataindex:
    """Test cases for dataindex validation."""
    
    def test_valid_dataindex(self):
        """Test that valid dataindex values pass validation."""
        # Test positive indices
        for i in range(len(DISPLACEMENT_DATA_LABELS)):
            validate_dataindex(i)  # Should not raise
        
        # Test negative indices (Python-style)
        for i in range(1, len(DISPLACEMENT_DATA_LABELS) + 1):
            validate_dataindex(-i)  # Should not raise
    
    def test_dataindex_negative_out_of_bounds(self):
        """Test that negative dataindex beyond valid range raises ValueError."""
        total_length = len(DISPLACEMENT_DATA_LABELS)
        with pytest.raises(ValueError, match=f"dataindex {-total_length - 1} is out of range"):
            validate_dataindex(-total_length - 1)
    
    def test_dataindex_too_large(self):
        """Test that dataindex larger than available labels raises ValueError."""
        max_valid = len(DISPLACEMENT_DATA_LABELS) - 1
        with pytest.raises(ValueError, match=f"dataindex {max_valid + 1} is out of range"):
            validate_dataindex(max_valid + 1)
    
    def test_dataindex_not_integer(self):
        """Test that non-integer dataindex raises ValueError."""
        with pytest.raises(ValueError, match="dataindex must be an integer"):
            validate_dataindex(3.14)
        
        with pytest.raises(ValueError, match="dataindex must be an integer"):
            validate_dataindex("3")
    
    def test_custom_max_index(self):
        """Test that custom max_index parameter works correctly."""
        validate_dataindex(2, max_index=5)  # Should pass
        validate_dataindex(-1, max_index=5)  # Should pass (negative indexing)
        
        with pytest.raises(ValueError, match="dataindex 6 is out of range"):
            validate_dataindex(6, max_index=5)
        
        with pytest.raises(ValueError, match="dataindex -7 is out of range"):
            validate_dataindex(-7, max_index=5)


class TestValidateFileList:
    """Test cases for file list validation."""
    
    def test_valid_file_list(self):
        """Test that valid file lists pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, "test1.dat")
            file2 = os.path.join(tmpdir, "test2.dat")
            
            with open(file1, 'w') as f:
                f.write("test data")
            with open(file2, 'w') as f:
                f.write("test data")
            
            validate_file_list([file1, file2])  # Should not raise
    
    def test_empty_file_list(self):
        """Test that empty file list raises ValueError."""
        with pytest.raises(ValueError, match="file_list cannot be empty"):
            validate_file_list([])
    
    def test_non_existent_files(self):
        """Test that non-existent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="The following files were not found"):
            validate_file_list(["/nonexistent/file1.dat", "/nonexistent/file2.dat"])
    
    def test_mixed_existing_nonexistent(self):
        """Test handling of mixed existing and non-existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, "exists.dat")
            with open(existing_file, 'w') as f:
                f.write("test")
            
            with pytest.raises(FileNotFoundError, match="nonexistent.dat"):
                validate_file_list([existing_file, "/nonexistent.dat"])
    
    def test_invalid_file_list_type(self):
        """Test that invalid file list types raise ValueError."""
        with pytest.raises(ValueError, match="file_list must be a list or tuple"):
            validate_file_list("not_a_list")
        
        with pytest.raises(ValueError, match="All file paths must be strings"):
            validate_file_list([123, 456])


class TestValidateLoopParameters:
    """Test cases for loop parameter validation."""
    
    def test_valid_loop_parameters(self):
        """Test that valid loop parameters pass validation."""
        validate_loop_parameters(0, 10)
        validate_loop_parameters(5, 5)  # Equal values should be allowed
        validate_loop_parameters(100, 200)
    
    def test_negative_loop_start(self):
        """Test that negative loop_start raises ValueError."""
        with pytest.raises(ValueError, match="loop_start must be non-negative"):
            validate_loop_parameters(-1, 10)
    
    def test_negative_loop_end(self):
        """Test that negative loop_end raises ValueError."""
        with pytest.raises(ValueError, match="loop_end must be non-negative"):
            validate_loop_parameters(0, -1)
    
    def test_loop_start_greater_than_end(self):
        """Test that loop_start > loop_end raises ValueError."""
        with pytest.raises(ValueError, match="loop_start \\(10\\) must be less than or equal to loop_end \\(5\\)"):
            validate_loop_parameters(10, 5)
    
    def test_non_integer_parameters(self):
        """Test that non-integer parameters raise ValueError."""
        with pytest.raises(ValueError, match="loop_start and loop_end must be integers"):
            validate_loop_parameters(3.14, 10)
        
        with pytest.raises(ValueError, match="loop_start and loop_end must be integers"):
            validate_loop_parameters(0, "10")


class TestValidateChunksParameter:
    """Test cases for chunks parameter validation."""
    
    def test_valid_chunks(self):
        """Test that valid chunk numbers pass validation."""
        validate_chunks_parameter(1)
        validate_chunks_parameter(50)
        validate_chunks_parameter(1000)
    
    def test_chunks_too_small(self):
        """Test that chunks smaller than minimum raises ValueError."""
        with pytest.raises(ValueError, match="nchunks must be at least 1"):
            validate_chunks_parameter(0)
    
    def test_chunks_too_large(self):
        """Test that chunks larger than maximum raises ValueError."""
        with pytest.raises(ValueError, match="nchunks cannot exceed 1000"):
            validate_chunks_parameter(1001)
    
    def test_custom_limits(self):
        """Test that custom min/max limits work correctly."""
        validate_chunks_parameter(5, min_chunks=5, max_chunks=10)
        
        with pytest.raises(ValueError, match="nchunks must be at least 5"):
            validate_chunks_parameter(3, min_chunks=5)
        
        with pytest.raises(ValueError, match="nchunks cannot exceed 10"):
            validate_chunks_parameter(15, max_chunks=10)
    
    def test_non_integer_chunks(self):
        """Test that non-integer chunks raise ValueError."""
        with pytest.raises(ValueError, match="nchunks must be an integer"):
            validate_chunks_parameter(3.14)


class TestExtractElementLabel:
    """Test cases for element label extraction from filenames."""
    
    def test_normal_filename(self):
        """Test extraction from normal filenames."""
        assert extract_element_label_from_filename("/path/to/Hf_data.dat") == "Hf"
        assert extract_element_label_from_filename("Ta_results.txt") == "Ta"
        assert extract_element_label_from_filename("O_analysis.dat") == "O_"
    
    def test_short_filename(self):
        """Test extraction from short filenames."""
        assert extract_element_label_from_filename("H") == "H"
        assert extract_element_label_from_filename("X.dat") == "X."
    
    def test_empty_filename(self):
        """Test extraction from empty filename."""
        assert extract_element_label_from_filename("") == "??"
    
    def test_path_with_filename(self):
        """Test that path is properly stripped."""
        assert extract_element_label_from_filename("/very/long/path/to/Al_test.dat") == "Al"


class TestInlinedValidationInPlotTimeseries:
    """Test cases for validation logic that was inlined into plot_displacement_timeseries."""
    
    def test_plot_displacement_timeseries_file_validation(self):
        """Test that file validation works in plot_displacement_timeseries."""
        from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_timeseries
        
        # Test empty file list
        with pytest.raises(ValueError, match="file_list cannot be empty"):
            plot_displacement_timeseries([], "test", 0, 5, 0, 10)
    
    def test_plot_displacement_timeseries_dataindex_validation(self):
        """Test that dataindex validation works in plot_displacement_timeseries.""" 
        from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_timeseries
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.dat")
            with open(test_file, 'w') as f:
                f.write("test")
            
            # Test non-integer dataindex
            with pytest.raises(ValueError, match="dataindex must be an integer"):
                plot_displacement_timeseries([test_file], "test", "not_int", 5, 0, 10)


if __name__ == "__main__":
    pytest.main([__file__])
