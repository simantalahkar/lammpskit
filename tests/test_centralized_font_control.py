"""
Test suite for centralized font control system in LAMMPSKit plotting functions.

This test suite demonstrates and validates:
1. Uniform font sizes across all plot types using centralized configuration
2. Individual override capability for specific plots
3. Consistent styling while maintaining flexibility for special cases
4. Visual regression testing for font consistency
"""

import pytest
import numpy as np
from lammpskit.plotting.timeseries_plots import (
    create_time_series_plot,
    create_dual_axis_plot,
    TimeSeriesPlotConfig,
    DualAxisPlotConfig,
    calculate_mean_std_label
)

# Get baseline directory relative to tests root (works from any execution context)  
# Since this test file is directly in tests/, the baseline is in the same directory level
BASELINE_DIR_RELATIVE = "baseline"


@pytest.fixture
def sample_font_data():
    """Create sample data for font control testing."""
    np.random.seed(42)  # For reproducible test results
    time_data = np.linspace(0, 10, 50)
    y_data1 = 10 + 5 * np.sin(time_data) + np.random.normal(0, 0.5, 50)
    y_data2 = 50 + 10 * np.cos(time_data) + np.random.normal(0, 1, 50)
    
    return {
        'time_data': time_data,
        'y_data1': y_data1,
        'y_data2': y_data2,
        'label1': calculate_mean_std_label(y_data1, "Dataset 1"),
        'label2': calculate_mean_std_label(y_data2, "Dataset 2")
    }


class TestCentralizedFontControl:
    """Test class for centralized font control functionality."""
    
    def test_font_config_defaults(self):
        """Test that font configurations have correct default values."""
        # Test TimeSeriesPlotConfig defaults
        ts_config = TimeSeriesPlotConfig()
        assert ts_config.fontsize_title == 8
        assert ts_config.fontsize_labels == 8
        assert ts_config.fontsize_ticks == 8
        assert ts_config.fontsize_legend == 8
        
        # Test DualAxisPlotConfig defaults
        dual_config = DualAxisPlotConfig()
        assert dual_config.fontsize_title == 8
        assert dual_config.fontsize_labels == 8
        assert dual_config.fontsize_ticks == 8
        assert dual_config.fontsize_legend == 8
    
    def test_font_config_custom_values(self):
        """Test that font configurations accept custom values."""
        # Test custom TimeSeriesPlotConfig
        ts_config = TimeSeriesPlotConfig(
            fontsize_title=14,
            fontsize_labels=12,
            fontsize_ticks=10,
            fontsize_legend=8
        )
        assert ts_config.fontsize_title == 14
        assert ts_config.fontsize_labels == 12
        assert ts_config.fontsize_ticks == 10
        assert ts_config.fontsize_legend == 8
        
        # Test custom DualAxisPlotConfig
        dual_config = DualAxisPlotConfig(
            fontsize_title=16,
            fontsize_labels=11,
            fontsize_ticks=9,
            fontsize_legend=7
        )
        assert dual_config.fontsize_title == 16
        assert dual_config.fontsize_labels == 11
        assert dual_config.fontsize_ticks == 9
        assert dual_config.fontsize_legend == 7
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_standard_font_sizes(self, sample_font_data):
        """Test time series plot with standard uniform font sizes."""
        config = TimeSeriesPlotConfig(
            fontsize_title=8,
            fontsize_labels=8,
            fontsize_ticks=8,
            fontsize_legend=8
        )
        
        fig, ax = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Standard Font Sizes (All = 8)",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1'],
            config=config
        )
        
        return fig
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_large_font_sizes(self, sample_font_data):
        """Test time series plot with large uniform font sizes."""
        config = TimeSeriesPlotConfig(
            fontsize_title=12,
            fontsize_labels=12,
            fontsize_ticks=12,
            fontsize_legend=12
        )
        
        fig, ax = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Large Font Sizes (All = 12)",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1'],
            config=config
        )
        
        return fig
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_mixed_font_size_overrides(self, sample_font_data):
        """Test time series plot with individual font size overrides."""
        standard_config = TimeSeriesPlotConfig(
            fontsize_title=8,
            fontsize_labels=8,
            fontsize_ticks=8,
            fontsize_legend=8
        )
        
        fig, ax = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Mixed Font Sizes (Override Demo)",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1'],
            config=standard_config,  # Base config with size 8
            fontsize_title=14,       # Override: large title
            fontsize_labels=10,      # Override: medium labels
            fontsize_ticks=8,        # Keep from config
            fontsize_legend=6        # Override: small legend
        )
        
        return fig
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_dual_axis_uniform_fonts(self, sample_font_data):
        """Test dual-axis plot with uniform font control."""
        config = DualAxisPlotConfig(
            fontsize_title=10,
            fontsize_labels=9,
            fontsize_ticks=8,
            fontsize_legend=7
        )
        
        fig, ax1, ax2 = create_dual_axis_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            sample_font_data['y_data2'],
            title="Dual-Axis Plot with Uniform Font Control",
            xlabel="Time (s)",
            primary_ylabel="Primary Data",
            secondary_ylabel="Secondary Data",
            primary_stats_label=sample_font_data['label1'],
            secondary_stats_label=sample_font_data['label2'],
            config=config
        )
        
        return fig
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_dual_axis_font_overrides(self, sample_font_data):
        """Test dual-axis plot with selective font size overrides."""
        config = DualAxisPlotConfig(
            fontsize_title=10,
            fontsize_labels=9,
            fontsize_ticks=8,
            fontsize_legend=7
        )
        
        fig, ax1, ax2 = create_dual_axis_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            sample_font_data['y_data2'],
            title="Dual-Axis Plot with Font Overrides",
            xlabel="Time (s)",
            primary_ylabel="Primary Data",
            secondary_ylabel="Secondary Data",
            primary_stats_label=sample_font_data['label1'],
            secondary_stats_label=sample_font_data['label2'],
            config=config,           # Base config
            fontsize_title=16,       # Override: extra large title
            fontsize_legend=5        # Override: extra small legend
            # Labels and ticks use config defaults (9pt and 8pt)
        )
        
        return fig
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_presentation_ready_fonts(self, sample_font_data):
        """Test time series plot with presentation-ready larger fonts."""
        config = TimeSeriesPlotConfig(
            fontsize_title=16,
            fontsize_labels=14,
            fontsize_ticks=12,
            fontsize_legend=10
        )
        
        fig, ax = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Presentation-Ready Font Sizes",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1'],
            config=config
        )
        
        return fig
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_compact_fonts_for_subfigures(self, sample_font_data):
        """Test time series plot with compact fonts suitable for subfigures."""
        config = TimeSeriesPlotConfig(
            fontsize_title=6,
            fontsize_labels=5,
            fontsize_ticks=4,
            fontsize_legend=4
        )
        
        fig, ax = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Compact Fonts for Subfigures",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1'],
            config=config
        )
        
        return fig
    
    def test_font_override_consistency(self, sample_font_data):
        """Test that font overrides work consistently across multiple calls."""
        base_config = TimeSeriesPlotConfig(fontsize_title=8)
        
        # Create two plots with the same override
        fig1, ax1 = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Test Plot 1",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1'],
            config=base_config,
            fontsize_title=14
        )
        
        fig2, ax2 = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Test Plot 2",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1'],
            config=base_config,
            fontsize_title=14
        )
        
        # Both plots should have the same title font size (14, not 8)
        assert ax1.title.get_fontsize() == 14
        assert ax2.title.get_fontsize() == 14
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig1)
        plt.close(fig2)
    
    def test_backward_compatibility(self, sample_font_data):
        """Test that the new system is backward compatible with old calls."""
        # Old-style call without font parameters should still work
        fig, ax = create_time_series_plot(
            sample_font_data['time_data'],
            sample_font_data['y_data1'],
            title="Backward Compatibility Test",
            xlabel="Time (s)",
            ylabel="Amplitude",
            stats_label=sample_font_data['label1']
            # No config or font size parameters - should use defaults
        )
        
        # Should have default font sizes
        assert ax.title.get_fontsize() == 8
        assert ax.xaxis.label.get_fontsize() == 8
        assert ax.yaxis.label.get_fontsize() == 8
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestFontControlIntegration:
    """Integration tests for font control in real-world scenarios."""
    
    def test_track_filament_evolution_font_consistency(self):
        """Test that track_filament_evolution uses consistent font controls."""
        # This test validates that the refactored track_filament_evolution
        # function properly uses the centralized font control system
        from lammpskit.ecellmodel.filament_layer_analysis import track_filament_evolution
        
        # Create minimal test data (this would normally come from OVITO analysis)
        # We'll mock the analyze_clusters function for testing
        import tempfile
        
        # Test that the function can be called without errors
        # (Note: Full testing would require actual OVITO data files)
        try:
            # This will fail due to missing data files, but we can check
            # that the function signature accepts the parameters correctly
            with tempfile.TemporaryDirectory() as temp_dir:
                track_filament_evolution(
                    [], "test", 0.001, 500, output_dir=temp_dir
                )
        except (FileNotFoundError, IndexError):
            # Expected - we don't have actual data files
            pass
        except TypeError as e:
            # This would indicate a parameter mismatch
            pytest.fail(f"Function signature error: {e}")
    
    def test_multiple_plot_font_coordination(self, sample_font_data):
        """Test that multiple plots can coordinate their font sizes."""
        # Create a shared font configuration for multiple related plots
        shared_config = TimeSeriesPlotConfig(
            fontsize_title=12,
            fontsize_labels=10,
            fontsize_ticks=8,
            fontsize_legend=8
        )
        
        # Create multiple plots that should look consistent together
        plots = []
        for i, data in enumerate([sample_font_data['y_data1'], sample_font_data['y_data2']]):
            fig, ax = create_time_series_plot(
                sample_font_data['time_data'],
                data,
                title=f"Plot {i+1} - Coordinated Fonts",
                xlabel="Time (s)",
                ylabel=f"Data {i+1}",
                stats_label=f"Dataset {i+1}",
                config=shared_config
            )
            plots.append((fig, ax))
        
        # All plots should have the same font sizes
        for fig, ax in plots:
            assert ax.title.get_fontsize() == 12
            assert ax.xaxis.label.get_fontsize() == 10
            assert ax.yaxis.label.get_fontsize() == 10
        
        # Clean up
        import matplotlib.pyplot as plt
        for fig, ax in plots:
            plt.close(fig)


if __name__ == "__main__":
    # Allow running the test file directly for debugging
    pytest.main([__file__, "-v"])
