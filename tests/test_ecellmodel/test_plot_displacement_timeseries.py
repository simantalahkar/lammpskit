import os
import glob
import tempfile
import pytest
from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_timeseries

@pytest.mark.parametrize("dataindex", range(-6,0))
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_displacement_timeseries(tmp_path, dataindex):
    # Prepare file list and labels
    data_dir = os.path.join(os.path.dirname(__file__), "test_data", "data_for_timeseries")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[1-9][A-Z][a-z]mobilestc1.dat")))
    print('data directory path to file_list',data_dir)
    datatype = "mobility"
    Nchunks = 12
    loop_start = 1
    loop_end = 100

    # Call the function
    figs = plot_displacement_timeseries(
        file_list=file_list,
        datatype=datatype,
        dataindex=dataindex,
        Nchunks=Nchunks,
        loop_start=loop_start,
        loop_end=loop_end,
        output_dir=tmp_path
    )
    return figs["displacement_timeseries"]


# ============================================================================
# NEW VALIDATION TESTS FOR plot_displacement_timeseries() ERROR PATHS
# ============================================================================

class TestPlotDisplacementTimeseriesValidation:
    """
    Test cases for validation logic in plot_displacement_timeseries().
    These tests verify the inline validation code that checks parameters
    before the function performs its main plotting operations.
    """

    def setup_method(self):
        """Set up valid test data for validation tests."""
        self.data_dir = os.path.join(os.path.dirname(__file__), "test_data", "data_for_timeseries")
        self.valid_file_list = sorted(glob.glob(os.path.join(self.data_dir, "[1-9][A-Z][a-z]mobilestc1.dat")))
        self.valid_datatype = "mobility"
        self.valid_dataindex = -1  # Use a valid index for testing
        self.valid_nchunks = 12
        self.valid_loop_start = 1
        self.valid_loop_end = 5  # Use smaller range for faster tests

    # File List Validation Tests
    def test_plot_displacement_timeseries_empty_file_list(self, tmp_path):
        """Test that empty file list raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=[],
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "file_list cannot be empty" in str(excinfo.value)

    def test_plot_displacement_timeseries_invalid_file_list_type(self, tmp_path):
        """Test that non-list file_list raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list="not_a_list",
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "file_list must be a list" in str(excinfo.value)

    def test_plot_displacement_timeseries_non_string_files(self, tmp_path):
        """Test that non-string items in file_list raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=[self.valid_file_list[0], 123, self.valid_file_list[1]],
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "All items in file_list must be strings" in str(excinfo.value)

    def test_plot_displacement_timeseries_missing_files(self, tmp_path):
        """Test that non-existent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as excinfo:
            plot_displacement_timeseries(
                file_list=[self.valid_file_list[0], "nonexistent_file.dat"],
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "File not found: nonexistent_file.dat" in str(excinfo.value)

    # Data Index Validation Tests
    def test_plot_displacement_timeseries_invalid_dataindex_type(self, tmp_path):
        """Test that non-integer dataindex raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex="not_an_integer",
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "dataindex must be an integer" in str(excinfo.value)

    def test_plot_displacement_timeseries_dataindex_float(self, tmp_path):
        """Test that float dataindex raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=1.5,
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "dataindex must be an integer" in str(excinfo.value)

    # Loop Parameters Validation Tests
    def test_plot_displacement_timeseries_invalid_loop_start_type(self, tmp_path):
        """Test that non-integer loop_start raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start="not_an_integer",
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "loop_start and loop_end must be integers" in str(excinfo.value)

    def test_plot_displacement_timeseries_invalid_loop_end_type(self, tmp_path):
        """Test that non-integer loop_end raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=5.5,
                output_dir=tmp_path
            )
        assert "loop_start and loop_end must be integers" in str(excinfo.value)

    def test_plot_displacement_timeseries_negative_loop_start(self, tmp_path):
        """Test that negative loop_start raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=-1,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "loop_start must be non-negative" in str(excinfo.value)

    def test_plot_displacement_timeseries_negative_loop_end(self, tmp_path):
        """Test that negative loop_end raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=self.valid_loop_start,
                loop_end=-1,
                output_dir=tmp_path
            )
        assert "loop_end must be non-negative" in str(excinfo.value)

    def test_plot_displacement_timeseries_loop_start_greater_than_end(self, tmp_path):
        """Test that loop_start > loop_end raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=self.valid_nchunks,
                loop_start=10,
                loop_end=5,
                output_dir=tmp_path
            )
        assert "loop_start must be less than or equal to loop_end" in str(excinfo.value)

    # Nchunks Parameter Validation Tests
    def test_plot_displacement_timeseries_invalid_nchunks_type(self, tmp_path):
        """Test that non-integer Nchunks raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks="not_an_integer",
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "Nchunks must be an integer" in str(excinfo.value)

    def test_plot_displacement_timeseries_nchunks_too_small(self, tmp_path):
        """Test that Nchunks < 1 raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=0,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "Nchunks must be between 1 and 100" in str(excinfo.value)

    def test_plot_displacement_timeseries_nchunks_too_large(self, tmp_path):
        """Test that Nchunks > 100 raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=101,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "Nchunks must be between 1 and 100" in str(excinfo.value)

    def test_plot_displacement_timeseries_nchunks_float(self, tmp_path):
        """Test that float Nchunks raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            plot_displacement_timeseries(
                file_list=self.valid_file_list,
                datatype=self.valid_datatype,
                dataindex=self.valid_dataindex,
                Nchunks=12.5,
                loop_start=self.valid_loop_start,
                loop_end=self.valid_loop_end,
                output_dir=tmp_path
            )
        assert "Nchunks must be an integer" in str(excinfo.value)

    # File Processing Error Tests
    def test_plot_displacement_timeseries_malformed_file(self, tmp_path):
        """Test that malformed displacement data files raise ValueError."""
        # Create a temporary malformed file
        with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.dat') as tmp_file:
            tmp_file.write("This is not a valid displacement data file\n")
            tmp_file.write("It lacks the proper format\n")
            tmp_file.flush()
            malformed_file = tmp_file.name

        try:
            with pytest.raises(ValueError) as excinfo:
                plot_displacement_timeseries(
                    file_list=[malformed_file],
                    datatype=self.valid_datatype,
                    dataindex=self.valid_dataindex,
                    Nchunks=self.valid_nchunks,
                    loop_start=0,
                    loop_end=1,
                    output_dir=tmp_path
                )
            assert "Failed to process file" in str(excinfo.value)
        finally:
            # Clean up the temporary file
            os.unlink(malformed_file)

    def test_plot_displacement_timeseries_valid_parameters_successful(self, tmp_path):
        """Test that valid parameters produce successful execution (sanity check)."""
        # This test ensures our validation doesn't break valid usage
        figs = plot_displacement_timeseries(
            file_list=self.valid_file_list[:2],  # Use first 2 files for speed
            datatype=self.valid_datatype,
            dataindex=self.valid_dataindex,
            Nchunks=self.valid_nchunks,
            loop_start=self.valid_loop_start,
            loop_end=self.valid_loop_end,
            output_dir=tmp_path
        )
        assert "displacement_timeseries" in figs
        assert figs["displacement_timeseries"] is not None

# ============================================================================
# END OF NEW VALIDATION TESTS FOR plot_displacement_timeseries()
# ============================================================================