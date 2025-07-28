import numpy as np
import pytest
import warnings
from pathlib import Path
from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters

# Get the directory containing this test file
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data" / "data_for_layer_analysis"

def test_analyze_clusters_output_values():
    filepath = str(TEST_DATA_DIR / "3set-0.1V.500000.lammpstrj")
    result = analyze_clusters(filepath)
    assert isinstance(result, tuple)
    assert len(result) == 10

    timestep, connection, filament_size_down, filament_height, rdf_down, filament_size_up, filament_depth, rdf_up, separation, gap = result

    assert timestep == 500000
    assert connection == 1
    assert filament_size_down == 214
    assert round(filament_height, 4) == 17.8633
    assert isinstance(rdf_down, np.ndarray)
    assert rdf_down.ndim == 2 and rdf_down.shape[0] > 0 and rdf_down.shape[1] > 0
    assert filament_size_up == 552
    assert round(filament_depth, 4) == 12.5072
    assert isinstance(rdf_up, np.ndarray)
    assert rdf_up.ndim == 2 and rdf_up.shape[0] > 0 and rdf_up.shape[1] > 0
    assert separation == 0
    assert gap == 0

def test_analyze_clusters_missing_file():
    missing_path = str(TEST_DATA_DIR / "nonexistent_file.lammpstrj")
    with pytest.raises(FileNotFoundError) as excinfo:
        analyze_clusters(missing_path)
    assert "File not found" in str(excinfo.value)

def test_analyze_clusters_malformed_file():
    # Create a temporary file with invalid content
    filepath = str(TEST_DATA_DIR / "malformed-0.1V.500000.lammpstrj")
    with pytest.raises(ValueError) as excinfo:    
        analyze_clusters(filepath)
    assert "Malformed or unreadable file for OVITO" in str(excinfo.value)

def test_analyze_clusters_no_clusters():
    # This file should be a valid LAMMPS trajectory file with no atoms,
    # or only atoms that do not match the selection criteria.
    filepath = str(TEST_DATA_DIR / "no_clusters-0.1V.500000.lammpstrj")
    with pytest.raises(ValueError) as excinfo:
        analyze_clusters(filepath)
    assert "No clusters found in file" in str(excinfo.value)


# ============================================================================
# NEW VALIDATION TESTS FOR analyze_clusters() PARAMETER VALIDATION
# ============================================================================

class TestAnalyzeClustersParameterValidation:
    """
    Test cases for parameter validation logic in analyze_clusters().
    These tests verify the new validation code that checks parameter types,
    ranges, and relationships before the function performs cluster analysis.
    """

    def setup_method(self):
        """Set up valid test data for validation tests."""
        self.valid_filepath = str(TEST_DATA_DIR / "3set-0.1V.500000.lammpstrj")
        self.valid_z_lower = 5.0
        self.valid_z_upper = 23.0
        self.valid_thickness = 21.0

    # Filepath Parameter Type Validation Tests
    def test_analyze_clusters_invalid_filepath_type_int(self):
        """Test that integer filepath raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            analyze_clusters(
                filepath=123,
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit=self.valid_z_upper,
                thickness=self.valid_thickness
            )
        assert "filepath must be a string" in str(excinfo.value)

    def test_analyze_clusters_invalid_filepath_type_none(self):
        """Test that None filepath raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            analyze_clusters(
                filepath=None,
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit=self.valid_z_upper,
                thickness=self.valid_thickness
            )
        assert "filepath must be a string" in str(excinfo.value)

    def test_analyze_clusters_invalid_filepath_type_list(self):
        """Test that list filepath raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            analyze_clusters(
                filepath=['not', 'a', 'string'],
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit=self.valid_z_upper,
                thickness=self.valid_thickness
            )
        assert "filepath must be a string" in str(excinfo.value)

    def test_analyze_clusters_empty_filepath(self):
        """Test that empty string filepath raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            analyze_clusters(
                filepath="",
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit=self.valid_z_upper,
                thickness=self.valid_thickness
            )
        assert "filepath cannot be empty" in str(excinfo.value)

    # Numeric Parameters Type Validation Tests
    def test_analyze_clusters_invalid_z_lower_type(self):
        """Test that non-numeric z_filament_lower_limit raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit="not_numeric",
                z_filament_upper_limit=self.valid_z_upper,
                thickness=self.valid_thickness
            )
        assert "z_filament_lower_limit must be numeric (int or float)" in str(excinfo.value)

    def test_analyze_clusters_invalid_z_upper_type(self):
        """Test that non-numeric z_filament_upper_limit raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit="not_numeric",
                thickness=self.valid_thickness
            )
        assert "z_filament_upper_limit must be numeric (int or float)" in str(excinfo.value)

    def test_analyze_clusters_invalid_thickness_type(self):
        """Test that non-numeric thickness raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit=self.valid_z_upper,
                thickness="not_numeric"
            )
        assert "thickness must be numeric (int or float)" in str(excinfo.value)

    def test_analyze_clusters_none_numeric_parameters(self):
        """Test that None values for numeric parameters raise TypeError."""
        with pytest.raises(TypeError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=None,
                z_filament_upper_limit=self.valid_z_upper,
                thickness=self.valid_thickness
            )
        assert "z_filament_lower_limit must be numeric (int or float)" in str(excinfo.value)

    # Parameter Range Validation Tests (Errors)
    def test_analyze_clusters_z_lower_greater_than_upper(self):
        """Test that z_lower > z_upper raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=25.0,
                z_filament_upper_limit=20.0,
                thickness=self.valid_thickness
            )
        assert "z_filament_lower_limit (25.0) must be less than z_filament_upper_limit (20.0)" in str(excinfo.value)

    def test_analyze_clusters_z_lower_equal_to_upper(self):
        """Test that z_lower == z_upper raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=20.0,
                z_filament_upper_limit=20.0,
                thickness=self.valid_thickness
            )
        assert "z_filament_lower_limit (20.0) must be less than z_filament_upper_limit (20.0)" in str(excinfo.value)

    def test_analyze_clusters_negative_thickness(self):
        """Test that negative thickness raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit=self.valid_z_upper,
                thickness=-5.0
            )
        assert "thickness (-5.0) must be positive" in str(excinfo.value)

    def test_analyze_clusters_zero_thickness(self):
        """Test that zero thickness raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=self.valid_z_lower,
                z_filament_upper_limit=self.valid_z_upper,
                thickness=0.0
            )
        assert "thickness (0.0) must be positive" in str(excinfo.value)

    # Parameter Range Validation Tests (Warnings)
    def test_analyze_clusters_negative_z_lower_warning(self):
        """Test that negative z_filament_lower_limit generates warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                analyze_clusters(
                    filepath=self.valid_filepath,
                    z_filament_lower_limit=-5.0,
                    z_filament_upper_limit=self.valid_z_upper,
                    thickness=self.valid_thickness
                )
            except (FileNotFoundError, ValueError):
                pass  # Expected to fail later, we're testing the warning
            
            # Check that warning was issued
            assert len(w) >= 1
            assert "z_filament_lower_limit (-5.0) is negative" in str(w[0].message)
            assert "coordinate system issues" in str(w[0].message)

    def test_analyze_clusters_large_z_values_warning(self):
        """Test that large z-coordinate values generate warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                analyze_clusters(
                    filepath=self.valid_filepath,
                    z_filament_lower_limit=2000.0,
                    z_filament_upper_limit=3000.0,
                    thickness=self.valid_thickness
                )
            except (FileNotFoundError, ValueError):
                pass  # Expected to fail later, we're testing the warning
            
            # Check that warning was issued
            assert len(w) >= 1
            assert "Large z-coordinate values detected" in str(w[0].message)
            assert "z_lower=2000.0, z_upper=3000.0" in str(w[0].message)
            assert "unit scale issues" in str(w[0].message)

    def test_analyze_clusters_large_negative_z_values_warning(self):
        """Test that large negative z-coordinate values generate warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                analyze_clusters(
                    filepath=self.valid_filepath,
                    z_filament_lower_limit=-2000.0,
                    z_filament_upper_limit=-1500.0,
                    thickness=self.valid_thickness
                )
            except (FileNotFoundError, ValueError):
                pass  # Expected to fail later, we're testing the warning
            
            # Check that warnings were issued
            assert len(w) >= 2  # Should have both negative and large value warnings
            warning_messages = [str(warning.message) for warning in w]
            
            # Check for negative z warning
            negative_warning = any("is negative" in msg for msg in warning_messages)
            assert negative_warning, f"Expected negative z warning, got: {warning_messages}"
            
            # Check for large z warning
            large_warning = any("Large z-coordinate values detected" in msg for msg in warning_messages)
            assert large_warning, f"Expected large z warning, got: {warning_messages}"

    # Edge Cases and Successful Validation Tests
    def test_analyze_clusters_very_small_z_difference(self):
        """Test that very small but positive z difference works (edge case)."""
        # This should not raise any errors or warnings, just test that validation passes
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                analyze_clusters(
                    filepath=self.valid_filepath,
                    z_filament_lower_limit=5.0,
                    z_filament_upper_limit=5.001,  # Very small difference
                    thickness=self.valid_thickness
                )
            except (FileNotFoundError, ValueError, Exception):
                pass  # Expected to continue to file processing
            
            # Should not generate warnings for normal small differences
            warning_messages = [str(warning.message) for warning in w]
            coordinate_warnings = [msg for msg in warning_messages if "coordinate" in msg or "scale" in msg]
            assert len(coordinate_warnings) == 0, f"Unexpected warnings for small valid difference: {coordinate_warnings}"

    def test_analyze_clusters_valid_integer_parameters(self):
        """Test that integer parameters are accepted (should work like floats)."""
        try:
            analyze_clusters(
                filepath=self.valid_filepath,
                z_filament_lower_limit=5,  # int instead of float
                z_filament_upper_limit=23, # int instead of float
                thickness=21               # int instead of float
            )
        except (FileNotFoundError, ValueError):
            pass  # Expected to fail on file processing, not parameter validation
        except TypeError:
            pytest.fail("Integer parameters should be accepted as valid numeric types")

    def test_analyze_clusters_validation_preserves_original_behavior(self):
        """Test that parameter validation doesn't interfere with original successful execution."""
        # This test uses the same parameters as the original successful test
        result = analyze_clusters(self.valid_filepath)
        assert isinstance(result, tuple)
        assert len(result) == 10
        # If we get here, validation didn't break the original functionality

# ============================================================================
# END OF NEW VALIDATION TESTS FOR analyze_clusters()
# ============================================================================